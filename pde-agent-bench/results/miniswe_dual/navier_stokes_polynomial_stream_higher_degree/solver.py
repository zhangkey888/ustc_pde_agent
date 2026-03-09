import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from dolfinx.nls import petsc as nls_petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    nu_val = case_spec.get("pde", {}).get("viscosity", 0.22)
    domain_spec = case_spec.get("domain", {})
    x_min = domain_spec.get("x_min", 0.0)
    x_max = domain_spec.get("x_max", 1.0)
    y_min = domain_spec.get("y_min", 0.0)
    y_max = domain_spec.get("y_max", 1.0)

    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)

    # P4/P3 Taylor-Hood with moderate mesh - high accuracy for polynomial solutions
    N = 16
    degree_u = 4
    degree_p = 3

    # Create mesh
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)

    gdim = domain.geometry.dim

    # Mixed element (Taylor-Hood P4/P3)
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)

    # Individual spaces for BCs
    V = fem.functionspace(domain, ("Lagrange", degree_u, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution
    u1_exact = x[0]**2 * (1 - x[0])**2 * (1 - 2*x[1])
    u2_exact = -2 * x[0] * (1 - x[0]) * (1 - 2*x[0]) * x[1] * (1 - x[1])
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    p_exact = x[0] + x[1]

    # Compute source term f = u·∇u - ν∇²u + ∇p
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.dot(grad_u_exact, u_exact)
    laplacian_exact = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)

    f = convection_exact - nu * laplacian_exact + grad_p_exact

    # Weak form (residual)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Velocity BC: u = u_exact on all boundaries
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.stack([
        X[0]**2 * (1 - X[0])**2 * (1 - 2*X[1]),
        -2 * X[0] * (1 - X[0]) * (1 - 2*X[0]) * X[1] * (1 - X[1])
    ], axis=0))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at origin to remove nullspace
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda X: X[0] + X[1])
    dofs_p = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], x_min) & np.isclose(X[1], y_min)
    )
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))

    bcs = [bc_u, bc_p]

    # Set initial guess: interpolate exact solution for fast convergence
    w.sub(0).interpolate(lambda X: np.stack([
        X[0]**2 * (1 - X[0])**2 * (1 - 2*X[1]),
        -2 * X[0] * (1 - X[0]) * (1 - 2*X[0]) * X[1] * (1 - X[1])
    ], axis=0))
    w.sub(1).interpolate(lambda X: X[0] + X[1])

    # Newton solver
    problem = fem_petsc.NewtonSolverNonlinearProblem(F_form, w, bcs=bcs)
    solver = nls_petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 30

    # Configure linear solver - use MUMPS for saddle-point system
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Evaluate on output grid
    xs = np.linspace(x_min, x_max, nx_out)
    ys = np.linspace(y_min, y_max, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }

    return result


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "viscosity": 0.22,
        },
        "domain": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6e}, {np.nanmax(result['u']):.6e}]")
    print(f"Solver info: {result['solver_info']}")

    # Compute error against exact solution
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    u1_ex = XX**2 * (1 - XX)**2 * (1 - 2*YY)
    u2_ex = -2 * XX * (1 - XX) * (1 - 2*XX) * YY * (1 - YY)
    vel_mag_exact = np.sqrt(u1_ex**2 + u2_ex**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2)) / (np.sqrt(np.mean(vel_mag_exact**2)) + 1e-15)
    print(f"Relative L2 error: {error:.6e}")
