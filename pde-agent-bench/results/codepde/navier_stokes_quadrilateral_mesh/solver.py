import numpy as np
from dolfinx import mesh, fem, default_scalar_type, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    nu_val = pde_config.get("viscosity", 0.1)
    nx_eval = pde_config.get("nx_eval", 50)
    ny_eval = pde_config.get("ny_eval", 50)

    # 2. Create mesh - quadrilateral as specified
    N = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function space (Taylor-Hood: Q2/Q1 for quads)
    degree_u = 2
    degree_p = 1

    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element manually using ufl
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)

    # 4. Define manufactured solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Exact solution
    u_exact = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                              -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    p_exact = ufl.as_vector([0.0])  # scalar zero

    # Compute source term: f = u·∇u - ν ∇²u + ∇p
    # ∇²u for vector = div(grad(u)) component-wise
    # grad(u) is a 2x2 tensor, u·∇u = grad(u) * u
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (grad u) * u = u·∇u
    diffusion = ufl.div(ufl.grad(u_exact))  # Laplacian of u
    # p_exact = 0 so grad(p) = 0
    f_expr = convection - nu_val * diffusion

    # 5. Nonlinear problem setup
    w = fem.Function(W)
    (u_test, p_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)

    # Residual: ν * inner(grad(u), grad(v)) + inner(grad(u)*u, v) - p*div(v) + div(u)*q - inner(f,v)
    F = (
        nu * ufl.inner(ufl.grad(u_sol), ufl.grad(u_test)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u_sol), u_sol), u_test) * ufl.dx
        - p_sol * ufl.div(u_test) * ufl.dx
        + ufl.div(u_sol) * p_test * ufl.dx
        - ufl.inner(f_expr, u_test) * ufl.dx
    )

    # 6. Boundary conditions
    # Velocity BC on all boundaries
    u_bc_func = fem.Function(V)

    # Interpolate exact velocity
    u_exact_expr = fem.Expression(
        ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])]),
        V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)

    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Locate DOFs for velocity sub-space
    W_sub0, sub0_map = W.sub(0).collapse()
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    # Pin pressure at corner
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    # 7. Initial guess: interpolate exact solution as starting point for Newton
    w_sub0 = w.sub(0)
    w_sub0.interpolate(u_bc_func)
    w.x.scatter_forward()

    # 8. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 25

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()

    # 9. Extract velocity and compute magnitude on evaluation grid
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

    u_values = np.full((points.shape[1], domain.geometry.dim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.reshape(-1, domain.geometry.dim)

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(nx_eval, ny_eval)

    # Replace any NaN with 0 (shouldn't happen on single proc)
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    total_linear_iters = n_newton  # approximate; each Newton step has one linear solve

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }