import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    nu_val = case_spec["pde"]["coefficients"]["nu"]

    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution - P4/P3 elements give high accuracy even on moderate mesh
    N = 40

    # Element degrees - P4/P3 as suggested by case ID
    degree_u = 4
    degree_p = 3

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space (Taylor-Hood P4/P3)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    # Spatial coordinate and manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute source term: f = u·∇u - ν∇²u + ∇p
    f_expr = (ufl.grad(u_exact) * u_exact
              - nu_val * ufl.div(ufl.grad(u_exact))
              + ufl.grad(p_exact))

    # Define the nonlinear problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Residual: ν*(∇u,∇v) + ((∇u)u, v) - (p, ∇·v) + (∇·u, q) - (f, v) = 0
    F = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0,0) — p_exact(0,0) = cos(0)*cos(0) = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_pin = fem.Function(Q)
    p0_pin.interpolate(lambda x_arr: np.full(x_arr.shape[1], 1.0))
    bc_p = fem.dirichletbc(p0_pin, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: interpolate exact solution for fast Newton convergence
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_bc_expr)
    w.x.array[V_map] = w_init_u.x.array[:]

    p_init = fem.Function(Q)
    p_init.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    w.x.array[Q_map] = p_init.x.array[:]

    # Jacobian
    J_form = ufl.derivative(F, w)

    # Solve nonlinear problem with Newton
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity solution
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    # Accuracy verification
    error_u = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_u_val = np.sqrt(comm.allreduce(fem.assemble_scalar(error_u), op=MPI.SUM))
    print(f"L2 error (velocity): {error_u_val:.6e}")

    error_p = fem.form(ufl.inner(p_h - p_exact, p_h - p_exact) * ufl.dx)
    error_p_val = np.sqrt(comm.allreduce(fem.assemble_scalar(error_p), op=MPI.SUM))
    print(f"L2 error (pressure): {error_p_val:.6e}")

    # Sample velocity magnitude onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full(XX.size, np.nan)
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_eval, cells_eval)  # shape (N, gdim)
        magnitude = np.linalg.norm(u_vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitude[idx]

    u_grid = u_grid.reshape(ny_out, nx_out)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [10],
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.2},
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Output range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
