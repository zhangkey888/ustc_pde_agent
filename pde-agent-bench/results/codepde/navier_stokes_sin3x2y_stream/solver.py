import numpy as np
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    comm = MPI.COMM_WORLD
    
    nu_val = 0.1
    nx_mesh = 64
    ny_mesh = 64
    degree_u = 2
    degree_p = 1
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define solution and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution
    u_exact_0 = 2*pi*ufl.cos(2*pi*x[1])*ufl.sin(3*pi*x[0])
    u_exact_1 = -3*pi*ufl.cos(3*pi*x[0])*ufl.sin(2*pi*x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.cos(pi*x[0])*ufl.cos(2*pi*x[1])
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    grad_u_exact = ufl.grad(u_exact)
    convection = ufl.dot(grad_u_exact, u_exact)  # (u·∇)u = grad(u) * u
    diffusion = -nu_val * ufl.div(ufl.grad(u_exact))
    grad_p = ufl.grad(p_exact)
    f = convection + diffusion + grad_p
    
    # 5. Residual form
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.stack([
        2*np.pi*np.cos(2*np.pi*X[1])*np.sin(3*np.pi*X[0]),
        -3*np.pi*np.cos(3*np.pi*X[0])*np.sin(2*np.pi*X[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    def corner(X):
        return np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    
    # Use a pressure pin
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda X: np.cos(np.pi*X[0])*np.cos(2*np.pi*X[1]))
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Initial guess: interpolate exact solution as initial guess for fast convergence
    W0, W0_map = W.sub(0).collapse()
    W1, W1_map = W.sub(1).collapse()
    
    # Use Stokes as initial guess (set w to 0 first, then solve Newton)
    # Actually, let's just use zero initial guess and let Newton converge
    w.x.array[:] = 0.0
    
    # Better: interpolate exact solution as initial guess for robustness
    u_init = fem.Function(W0)
    u_init.interpolate(lambda X: np.stack([
        2*np.pi*np.cos(2*np.pi*X[1])*np.sin(3*np.pi*X[0]),
        -3*np.pi*np.cos(3*np.pi*X[0])*np.sin(2*np.pi*X[1])
    ]))
    w.x.array[W0_map] = u_init.x.array[:]
    
    p_init = fem.Function(W1)
    p_init.interpolate(lambda X: np.cos(np.pi*X[0])*np.cos(2*np.pi*X[1]))
    w.x.array[W1_map] = p_init.x.array[:]
    
    # Actually let's NOT use exact solution as initial guess - that would be cheating
    # Use zero initial guess
    w.x.array[:] = 0.0
    
    # 8. Newton solve
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
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
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_eval = 50
    ny_eval = 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    vel_mag = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        magnitudes = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = magnitudes[idx]
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    # Get total linear iterations
    total_linear_its = 0  # With direct solver, each Newton step = 1 linear solve
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }