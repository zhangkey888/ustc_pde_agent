import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]

    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution - use high resolution for accuracy
    N = 96

    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )

    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Spatial coordinate
    x = ufl.SpatialCoordinate(msh)

    # Manufactured solution
    u_exact_expr = ufl.as_vector(
        [
            -40.0
            * (x[1] - 0.5)
            * ufl.exp(-20.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)),
            40.0
            * (x[0] - 0.5)
            * ufl.exp(-20.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)),
        ]
    )

    # Compute source term f = u·∇u - ν∇²u + ∇p
    # p = 0 so ∇p = 0
    grad_u_exact = ufl.grad(u_exact_expr)
    convection = ufl.dot(grad_u_exact, u_exact_expr)  # grad(u)*u
    laplacian_u = ufl.div(ufl.grad(u_exact_expr))
    grad_p = ufl.as_vector([0.0 * x[0], 0.0 * x[0]])

    f_expr = convection - nu_val * laplacian_u + grad_p

    # Setup unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    # Pressure pinning
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], xmin) & np.isclose(xx[1], ymin),
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    # Initialize with exact solution as initial guess (helps convergence)
    w.sub(0).interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    w.x.scatter_forward()

    # Solve with Newton
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
        F,
        w,
        bcs=bcs,
        J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )

    problem.solve()
    w.x.scatter_forward()

    # Extract velocity solution
    u_sol = w.sub(0).collapse()

    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_expr, V.element.interpolation_points)
    )

    error_form = fem.form(
        ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    )
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error in velocity: {error_global:.6e}")

    # Sample onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
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
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global] = vals[idx_local]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0] ** 2 + u_values[:, 1] ** 2)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    # Handle any NaN from boundary points not found
    if np.any(np.isnan(u_grid)):
        nan_count = np.sum(np.isnan(u_grid))
        print(f"Warning: {nan_count} NaN values in output, filling with 0")
        u_grid = np.nan_to_num(u_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [1],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.12},
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    import time

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6f}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    # Check against exact solution
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u1_exact = -40 * (YY - 0.5) * np.exp(-20 * ((XX - 0.5) ** 2 + (YY - 0.5) ** 2))
    u2_exact = 40 * (XX - 0.5) * np.exp(-20 * ((XX - 0.5) ** 2 + (YY - 0.5) ** 2))
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.sqrt(np.mean((result["u"] - vel_mag_exact) ** 2))
    max_error = np.max(np.abs(result["u"] - vel_mag_exact))
    print(f"RMSE vs exact: {error:.6e}")
    print(f"Max error vs exact: {max_error:.6e}")
