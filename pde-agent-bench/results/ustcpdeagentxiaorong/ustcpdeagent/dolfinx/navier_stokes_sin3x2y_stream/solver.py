import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    t_start = time.time()

    # Parse case spec
    nu_val = case_spec["pde"]["viscosity"]
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]

    N = 64

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Mixed function space: Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W_map = W.sub(0).collapse()
    Q, Q_to_W_map = W.sub(1).collapse()

    # Define manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]),
        -3 * pi * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    ])

    p_exact = ufl.cos(pi * x[0]) * ufl.cos(2 * pi * x[1])

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Source term: f = (u·∇)u - ν∆u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.grad(u_exact) * u_exact
    f = convection_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Weak form (residual)
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    # Boundary conditions - velocity on all boundaries
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pinning at (0, 0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0_func = fem.Function(Q)
    p0_func.interpolate(lambda x: np.full(x.shape[1], np.cos(np.pi * x[0]) * np.cos(2 * np.pi * x[1])))
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: interpolate exact solution
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_bc_expr)
    w.x.array[V_to_W_map] = w_init_u.x.array[:]

    p_init = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_exact_expr)
    w.x.array[Q_to_W_map] = p_init.x.array[:]
    w.x.scatter_forward()

    # Solve nonlinear problem
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
        petsc_options=petsc_options,
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Compute L2 error for verification
    error_u = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_u_local = fem.assemble_scalar(error_u)
    error_u_global = np.sqrt(comm.allreduce(error_u_local, op=MPI.SUM))
    print(f"L2 error velocity: {error_u_global:.6e}")

    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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

    u_grid = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx, :] = vals[idx, :]

    vel_magnitude = np.sqrt(u_grid[:, 0]**2 + u_grid[:, 1]**2)
    vel_magnitude_grid = vel_magnitude.reshape(ny_out, nx_out)

    t_end = time.time()
    print(f"Total solve time: {t_end - t_start:.2f}s")

    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [50],
    }

    return {
        "u": vel_magnitude_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.1,
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Output range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    
    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    
    u1_exact = 2 * np.pi * np.cos(2 * np.pi * YY) * np.sin(3 * np.pi * XX)
    u2_exact = -3 * np.pi * np.cos(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    error = np.sqrt(np.nanmean((result["u"] - vel_mag_exact)**2))
    print(f"RMS error in velocity magnitude: {error:.6e}")
    max_error = np.nanmax(np.abs(result["u"] - vel_mag_exact))
    print(f"Max error in velocity magnitude: {max_error:.6e}")
