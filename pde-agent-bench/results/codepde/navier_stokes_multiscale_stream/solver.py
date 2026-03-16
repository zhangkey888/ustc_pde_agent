import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.12)
    nx_eval = pde_config.get("nx_eval", 50)
    ny_eval = pde_config.get("ny_eval", 50)

    # Mesh resolution and element degrees
    N = 48
    degree_u = 2
    degree_p = 1

    comm = MPI.COMM_WORLD

    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function space (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)

    # 4. Define manufactured solution symbolically
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact velocity
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + (3 * pi / 5) * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - (9 * pi / 10) * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])

    # Exact pressure
    p_exact = ufl.cos(2 * pi * x[0]) * ufl.cos(pi * x[1])

    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Compute source term from manufactured solution
    # f = u·∇u - ν∇²u + ∇p
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # 5. Define variational form
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # 6. Boundary conditions
    # Velocity BC on entire boundary
    u_bc_func = fem.Function(V)

    # Interpolate exact velocity onto V
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to remove nullspace
    # Find DOF closest to (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    p_bc_func = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    # 7. Initial guess: interpolate exact solution (helps Newton converge fast)
    # We can also start from zero or Stokes, but exact is best for MMS
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()

    u_init = fem.Function(V_sub)
    u_init.interpolate(fem.Expression(u_exact, V_sub.element.interpolation_points))

    p_init = fem.Function(Q_sub)
    p_init.interpolate(fem.Expression(p_exact, Q_sub.element.interpolation_points))

    # Set initial guess in w
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # 8. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()

    # 9. Extract velocity on evaluation grid
    # Create evaluation points
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()

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
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    # Evaluate velocity
    u_values = np.full((points.shape[1], domain.geometry.dim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    # Get total linear iterations
    total_linear_its = 0
    try:
        total_linear_its = ksp.getIterationNumber() * n_newton
    except:
        total_linear_its = n_newton  # LU gives 1 iter per Newton step

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": n_newton,  # LU direct solve: 1 linear iter per Newton step
            "nonlinear_iterations": [int(n_newton)],
        }
    }