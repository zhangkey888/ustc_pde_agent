import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.16)
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function space (Taylor-Hood P2/P1)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 2, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    
    # Create mixed element
    vel_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    pres_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    mixed_el = ufl.MixedElement([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define solution and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Manufactured solution
    # u_exact = [pi*tanh(6*(x-0.5))*cos(pi*y), -6*(1 - tanh(6*(x-0.5))**2)*sin(pi*y)]
    # p_exact = sin(pi*x)*cos(pi*y)
    
    tanh_arg = 6.0 * (x[0] - 0.5)
    tanh_val = ufl.tanh(tanh_arg)
    sech2_val = 1.0 - tanh_val**2
    
    u1_exact = ufl.pi * tanh_val * ufl.cos(ufl.pi * x[1])
    u2_exact = -6.0 * sech2_val * ufl.sin(ufl.pi * x[1])
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    # In weak form: F = ν*(grad(u), grad(v)) + (grad(u)*u, v) - (p, div(v)) + (div(u), q) - (f, v) = 0
    # where f makes the manufactured solution exact
    
    # Source term
    f_expr = (
        ufl.grad(u_exact) * u_exact
        - nu_val * ufl.div(ufl.grad(u_exact))
        + ufl.grad(p_exact)
    )
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    # Velocity BC
    V_sub, _ = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    
    u_bc_func.interpolate(lambda X: np.vstack([
        np.pi * np.tanh(6.0 * (X[0] - 0.5)) * np.cos(np.pi * X[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (X[0] - 0.5))**2) * np.sin(np.pi * X[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove null space
    # Find a DOF near (0,0) for pressure
    Q_sub, _ = W.sub(1).collapse()
    
    def corner(X):
        return np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.cos(np.pi * X[1]))
    
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q_sub), corner)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 6. Initial guess - interpolate exact solution (helps Newton converge)
    # Set initial guess to zero or close to exact
    w_sub_u = w.sub(0)
    w_sub_p = w.sub(1)
    
    # Better: use a Stokes solve or interpolate exact solution as initial guess
    # We'll interpolate the exact solution as initial guess for fast convergence
    u_init = fem.Function(V_sub)
    u_init.interpolate(lambda X: np.vstack([
        np.pi * np.tanh(6.0 * (X[0] - 0.5)) * np.cos(np.pi * X[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (X[0] - 0.5))**2) * np.sin(np.pi * X[1])
    ]))
    
    p_init = fem.Function(Q_sub)
    p_init.interpolate(lambda X: np.sin(np.pi * X[0]) * np.cos(np.pi * X[1]))
    
    # Copy initial guess into w
    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(p_init)
    w.x.scatter_forward()
    
    # 7. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
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
    
    # 8. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    n_points = points_3d.shape[1]
    vel_mag = np.full(n_points, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_eval, gdim)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Get total linear iterations
    total_linear_its = 0
    # Newton solver doesn't easily expose per-step iterations, estimate
    nonlinear_iterations = [int(n_newton)]
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": nonlinear_iterations,
        }
    }