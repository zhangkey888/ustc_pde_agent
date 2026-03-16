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

    # Mesh resolution and element degrees
    N = 64
    degree_u = 2
    degree_p = 1

    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Function spaces (Taylor-Hood P2/P1)
    V_el = ("Lagrange", degree_u, (domain.geometry.dim,))
    Q_el = ("Lagrange", degree_p)

    V = fem.functionspace(domain, V_el)
    Q = fem.functionspace(domain, Q_el)

    # Mixed function space
    from dolfinx.fem import Function
    from basix.ufl import mixed_element, element

    vel_elem = element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    pres_elem = element("Lagrange", domain.basix_cell(), degree_p)
    mel = mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mel)

    # 4. Define the manufactured solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact velocity: u = (2*pi*cos(2*pi*y)*sin(2*pi*x), -2*pi*cos(2*pi*x)*sin(2*pi*y))
    u_exact = ufl.as_vector([
        2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(2 * pi * x[0]),
        -2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    ])

    # Exact pressure: p = sin(2*pi*x)*cos(2*pi*y)
    p_exact = ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Compute source term: f = u·∇u - ν ∇²u + ∇p
    # Note: -ν ∇²u = -ν div(grad(u)), so f = (u·∇)u - ν ∇²u + ∇p
    # In weak form we'll handle this properly
    # f = (u_exact · ∇)u_exact - ν Δu_exact + ∇p_exact
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # 5. Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Residual: ν (∇u, ∇v) + ((u·∇)u, v) - (p, ∇·v) + (∇·u, q) - (f, v)
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # 6. Boundary conditions
    # Velocity BC on all boundaries
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.vstack([
        2 * np.pi * np.cos(2 * np.pi * X[1]) * np.sin(2 * np.pi * X[0]),
        -2 * np.pi * np.cos(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1])
    ]))

    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to remove nullspace
    # Find a vertex near (0, 0)
    def corner(X):
        return np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)

    # Pressure value at (0,0): sin(0)*cos(0) = 0
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda X: np.sin(2 * np.pi * X[0]) * np.cos(2 * np.pi * X[1]))

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    # 7. Initial guess: interpolate exact solution for better convergence
    w_sub0 = w.sub(0)
    w_sub1 = w.sub(1)

    # Create temporary functions for interpolation
    u_init = fem.Function(V)
    u_init.interpolate(lambda X: np.vstack([
        2 * np.pi * np.cos(2 * np.pi * X[1]) * np.sin(2 * np.pi * X[0]),
        -2 * np.pi * np.cos(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1])
    ]))

    p_init = fem.Function(Q)
    p_init.interpolate(lambda X: np.sin(2 * np.pi * X[0]) * np.cos(2 * np.pi * X[1]))

    # Use zero initial guess (or Stokes) - let's try with zero and see
    # Actually, for manufactured solution with good mesh, zero guess + Newton should work
    # But let's use a perturbation of exact to help convergence
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # Perturb slightly to avoid exact solution being the starting point
    # (Newton would converge in 1 iteration trivially)
    # Actually, let's start from zero to be honest
    w.x.array[:] = 0.0

    # 8. Newton solve
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
    points[0] = XX.ravel()
    points[1] = YY.ravel()

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
    vel_magnitude = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]

    u_grid = vel_magnitude.reshape((nx_eval, ny_eval))

    # Get total linear iterations
    total_linear_its = 0
    # We can't easily get per-Newton-step iterations from the high-level API
    # but we report what we can

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }