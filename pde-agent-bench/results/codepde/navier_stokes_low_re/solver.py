import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 1.0)
    nx_eval = pde_config.get("nx_eval", 50)
    ny_eval = pde_config.get("ny_eval", 50)

    # 2. Mesh resolution and element degrees
    N = 40  # mesh resolution
    degree_u = 2
    degree_p = 1

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
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
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Exact velocity: u = (pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y))
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])

    # Exact pressure: p = cos(pi*x)*cos(pi*y)
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Source term: f = u·∇u - ν ∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u) * u
    laplacian_u = ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    f = convection - nu_val * laplacian_u + grad_p

    # 5. Define variational form (nonlinear residual)
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # 6. Boundary conditions
    # All boundaries: u = u_exact
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)

    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(
        u_exact, V.element.interpolation_points
    )
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to remove nullspace
    # Find a vertex at (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    # Pin pressure DOF
    p_bc_func = fem.Function(Q)
    p_bc_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)

    # Use geometrical approach to pin pressure at corner
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), corner)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))

    bcs = [bc_u, bc_p]

    # 7. Initial guess: interpolate exact solution (helps convergence)
    # Interpolate velocity part
    W0_sub, W0_map = W.sub(0).collapse()
    w0_init = fem.Function(W0_sub)
    w0_init.interpolate(fem.Expression(u_exact, W0_sub.element.interpolation_points))
    w.x.array[W0_map] = w0_init.x.array

    W1_sub, W1_map = W.sub(1).collapse()
    w1_init = fem.Function(W1_sub)
    w1_init.interpolate(fem.Expression(p_exact, W1_sub.element.interpolation_points))
    w.x.array[W1_map] = w1_init.x.array

    # Actually, start from zero to test Newton properly, but with nu=1.0 (low Re), zero should work
    # Let's use a Stokes-like initial guess (zero velocity)
    # Actually for accuracy and speed, let's just start from zero
    w.x.array[:] = 0.0

    # 8. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()

    # 9. Extract velocity on evaluation grid
    # Collapse velocity subspace
    V_collapse, collapse_map = W.sub(0).collapse()
    u_sol = fem.Function(V_collapse)
    u_sol.x.array[:] = w.x.array[collapse_map]

    # Create evaluation points
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

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

    u_values = np.full((points.shape[1], domain.geometry.dim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    # Count linear iterations (approximate: Newton solver doesn't directly expose per-step linear iters easily)
    total_linear_iters = n_newton  # approximate

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