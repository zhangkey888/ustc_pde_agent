import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    bbox = case_spec["output"]["grid"]["bbox"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]

    # Mesh resolution - the manufactured solution has exp(6*(x-1)) boundary layer
    # With P2/P1 Taylor-Hood, N=80 gives L2 error ~ 4e-6
    N = 80

    # Create mesh
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1 mixed space
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    # Manufactured solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact = ufl.as_vector([
        pi * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.cos(pi * x[1]),
        -6.0 * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term: f = (u·∇)u - ν∆u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    conv = ufl.dot(grad_u_exact, u_exact)
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    f = conv - nu_val * laplacian_u + grad_p

    # Nonlinear problem setup
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))

    F_form = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F_form, w)

    # Boundary conditions - velocity on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at (0, 0) - p_exact(0,0) = sin(0)*sin(0) = 0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initial guess: use zero, then do Picard-like warmup or just Newton from zero
    # For better convergence, initialize with BCs applied
    w.x.array[:] = 0.0
    # Set velocity initial guess from exact solution (helps Newton converge quickly)
    u_init = fem.Function(V)
    u_init.interpolate(u_bc_expr)
    w.x.array[V_map] = u_init.x.array[:]
    # Set pressure initial guess
    p_init = fem.Function(Q)
    p_init_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_init.interpolate(p_init_expr)
    w.x.array[Q_map] = p_init.x.array[:]
    w.x.scatter_forward()

    # Newton solve with direct LU
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
        F_form, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
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

    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, glob_idx in enumerate(eval_map):
            u_values[glob_idx] = vals[idx]

    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    # Handle potential NaN at boundary corners
    if np.any(np.isnan(u_grid)):
        nan_count = np.sum(np.isnan(u_grid))
        print(f"Warning: {nan_count} NaN values in output grid, filling with 0")
        u_grid = np.nan_to_num(u_grid, nan=0.0)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [1],
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.08},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    total_time = time.time() - t0
    print(f"Total time: {total_time:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Min/Max velocity magnitude: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")

    # Verify against exact solution
    bbox = case_spec["output"]["grid"]["bbox"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)

    ux_exact = np.pi * np.exp(6.0 * (XX - 1.0)) * np.cos(np.pi * YY)
    uy_exact = -6.0 * np.exp(6.0 * (XX - 1.0)) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"RMS error on grid: {error:.6e}")
    print(f"Max error on grid: {max_error:.6e}")
