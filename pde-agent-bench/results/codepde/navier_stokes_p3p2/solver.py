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

    # Mesh resolution and element degrees for P3/P2 Taylor-Hood
    N = 32  # mesh resolution
    degree_u = 3
    degree_p = 2

    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Mixed function space (Taylor-Hood P3/P2)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)

    # 4. Define manufactured solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Exact solution
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute source term: f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    # -ν ∇²u = -ν div(grad(u))
    f = (ufl.grad(u_exact) * u_exact
         - nu * ufl.div(ufl.grad(u_exact))
         + ufl.grad(p_exact))

    # 5. Define nonlinear residual
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # 6. Boundary conditions
    # All boundary: u = u_exact
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)

    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at one point to remove nullspace
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    # Use geometrical approach for pressure pin
    dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))

    bcs = [bc_u, bc_p]

    # 7. Initial guess: interpolate exact solution
    w_sub0 = w.sub(0)
    w_sub1 = w.sub(1)

    # Create temporary functions for interpolation
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.stack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))

    p_init = fem.Function(Q)
    p_init.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

    # Set initial guess by interpolating into the mixed space
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # Perturb slightly so Newton actually does work (or just solve from zero)
    # Actually, let's start from zero to test Newton properly
    w.x.array[:] = 0.0
    # Re-apply BC values
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)

    # Actually use a Stokes-like initial guess: start from exact to ensure convergence
    # But let's try from a slightly perturbed state
    # For robustness, just start from exact - Newton should converge in ~1 iteration
    # Let's start from zero and let Newton converge
    w.x.array[:] = 0.0

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

    # Evaluate velocity (2 components)
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :2]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    # Get total linear iterations
    total_ksp_its = ksp.getIterationNumber()  # last solve iterations

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
            "iterations": int(n_newton),  # LU so 1 iter per Newton step
        }
    }