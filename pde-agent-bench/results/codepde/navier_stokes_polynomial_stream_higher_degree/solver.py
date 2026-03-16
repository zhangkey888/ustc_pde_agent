import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    nu_val = float(pde_config.get("viscosity", 0.22))

    # 2. Create mesh - use high resolution for accuracy
    N = 64
    degree_u = 3
    degree_p = 2

    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function spaces (Taylor-Hood P3/P2)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)

    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)

    # Exact velocity: u = [x^2*(1-x)^2*(1-2*y), -2*x*(1-x)*(1-2*x)*y*(1-y)]
    u_exact_0 = x[0]**2 * (1 - x[0])**2 * (1 - 2*x[1])
    u_exact_1 = -2 * x[0] * (1 - x[0]) * (1 - 2*x[0]) * x[1] * (1 - x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])

    # Exact pressure: p = x + y
    p_exact = x[0] + x[1]

    # Compute source term: f = u·∇u - ν∇²u + ∇p
    # ∇u is grad(u_exact), u·∇u = grad(u_exact)*u_exact
    grad_u = ufl.grad(u_exact)
    convection = grad_u * u_exact  # (u·∇)u
    diffusion = -nu_val * ufl.div(ufl.grad(u_exact))  # -ν∇²u
    grad_p = ufl.grad(p_exact)
    f = convection + diffusion + grad_p

    # 5. Setup nonlinear problem
    w = fem.Function(W)
    (u_test, q_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)

    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Residual: ν*(∇u,∇v) + ((u·∇)u, v) - (p, ∇·v) + (∇·u, q) - (f, v) = 0
    F = (
        nu_c * ufl.inner(ufl.grad(u_sol), ufl.grad(u_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, u_test) * ufl.dx
        - p_sol * ufl.div(u_test) * ufl.dx
        + ufl.div(u_sol) * q_test * ufl.dx
        - ufl.inner(f, u_test) * ufl.dx
    )

    # 6. Boundary conditions
    # Velocity BC on entire boundary
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.stack([
        X[0]**2 * (1 - X[0])**2 * (1 - 2*X[1]),
        -2 * X[0] * (1 - X[0]) * (1 - 2*X[0]) * X[1] * (1 - X[1])
    ], axis=0))

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to remove nullspace
    # Find DOF closest to (0,0)
    def corner_marker(X):
        return np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)

    # Pin pressure at corner
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda X: X[0] + X[1])  # p_exact = x + y

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner_marker)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    # 7. Initial guess: interpolate exact solution
    W0_sub, W0_map = W.sub(0).collapse()
    W1_sub, W1_map = W.sub(1).collapse()

    u_init = fem.Function(W0_sub)
    u_init.interpolate(lambda X: np.stack([
        X[0]**2 * (1 - X[0])**2 * (1 - 2*X[1]),
        -2 * X[0] * (1 - X[0]) * (1 - 2*X[0]) * X[1] * (1 - X[1])
    ], axis=0))
    w.x.array[W0_map] = u_init.x.array

    p_init = fem.Function(W1_sub)
    p_init.interpolate(lambda X: X[0] + X[1])
    w.x.array[W1_map] = p_init.x.array
    w.x.scatter_forward()

    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    n_iters, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_iters} iterations"
    w.x.scatter_forward()

    # 9. Extract velocity on evaluation grid
    nx_eval = 50
    ny_eval = 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.stack([XX.ravel(), YY.ravel(), np.zeros(nx_eval * ny_eval)], axis=0)

    # Extract velocity sub-function
    u_h = w.sub(0).collapse()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    n_pts = points_2d.shape[1]
    vel_mag = np.full(n_pts, np.nan)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)

    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(n_iters),
            "nonlinear_iterations": [int(n_iters)],
        }
    }