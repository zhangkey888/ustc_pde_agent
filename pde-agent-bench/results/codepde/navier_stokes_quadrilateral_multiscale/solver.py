import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.1)
    nx_eval = pde_config.get("nx_eval", 50)
    ny_eval = pde_config.get("ny_eval", 50)

    # Mesh resolution - need high resolution for multiscale solution
    N = 96
    degree_u = 2
    degree_p = 1

    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)

    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact velocity
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + pi * ufl.cos(4 * pi * x[1]) * ufl.sin(2 * pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - (pi / 2) * ufl.cos(2 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])

    # Exact pressure
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(2 * pi * x[1])

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Compute source term from manufactured solution
    # f = u·∇u - ν ∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u) * u
    diffusion = -nu * ufl.div(ufl.grad(u_exact))
    pressure_grad = ufl.grad(p_exact)
    f = convection + diffusion + pressure_grad

    # 5. Setup nonlinear problem
    w = fem.Function(W)
    (u_test, p_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)

    # Residual
    F = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(u_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, u_test) * ufl.dx
        - p_sol * ufl.div(u_test) * ufl.dx
        + ufl.div(u_sol) * p_test * ufl.dx
        - ufl.inner(f, u_test) * ufl.dx
    )

    # 6. Boundary conditions
    # Velocity BC on all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)

    # Interpolate exact velocity BC
    u_bc_func = fem.Function(V)

    # Create expression for exact velocity
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to fix the constant
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    # Use a pressure BC at one point
    p_bc_func = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner_point)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    # 7. Initial guess: interpolate exact solution as starting point for Newton
    # This helps convergence significantly
    w_sub0 = w.sub(0)
    w_sub1 = w.sub(1)

    # Create separate functions for interpolation
    u_init = fem.Function(V)
    u_init.interpolate(u_exact_expr)

    p_init = fem.Function(Q)
    p_init.interpolate(p_exact_expr)

    # Use a Stokes-like initial guess (zero velocity) or exact solution
    # For robustness, let's use a perturbation of exact to test Newton
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # Actually, let's start from zero to properly test Newton convergence
    w.x.array[:] = 0.0

    # 8. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)

    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    solver.relaxation_parameter = 1.0

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # First try with zero initial guess - if it fails, use exact solution as init
    try:
        n_iters, converged = solver.solve(w)
        if not converged:
            raise RuntimeError("Newton did not converge from zero init")
    except Exception:
        # Restart with exact solution as initial guess (slightly perturbed)
        w.sub(0).interpolate(u_init)
        w.sub(1).interpolate(p_init)
        # Perturb slightly
        w.x.array[:] *= 0.9
        n_iters, converged = solver.solve(w)

    w.x.scatter_forward()

    # 9. Extract velocity on evaluation grid
    u_h = w.sub(0).collapse()

    # Create evaluation points
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0] = X.ravel()
    points[1] = Y.ravel()

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    # Evaluate velocity
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :2]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    # Get linear solver iterations info
    total_linear_iters = 0
    try:
        total_linear_iters = ksp.getIterationNumber()
    except Exception:
        total_linear_iters = 0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": total_linear_iters,
            "nonlinear_iterations": [int(n_iters)],
        }
    }