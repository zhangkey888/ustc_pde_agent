import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.18)
    
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
    
    # 4. Define exact solution for BCs and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity
    u_exact_0 = 6.0 * (1.0 - ufl.tanh(6.0 * (x[1] - 0.5))**2) * ufl.sin(pi * x[0])
    u_exact_1 = -pi * ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.cos(pi * x[0])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    
    # Exact pressure
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute source term: f = u·∇u - ν∇²u + ∇p
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # 5. Define variational problem
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
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    
    # Interpolate exact velocity
    u_exact_expr = fem.Expression(
        u_exact, V.element.interpolation_points
    )
    u_bc_func.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a point, e.g., (0,0)
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
    
    # 7. Initial guess: interpolate exact solution (helps convergence)
    w_sub0 = w.sub(0)
    w_sub1 = w.sub(1)
    
    # Set initial guess to exact solution for better convergence
    u_init = fem.Function(V)
    u_init.interpolate(u_exact_expr)
    
    p_init = fem.Function(Q)
    p_init.interpolate(p_exact_expr)
    
    # We need to set initial guess in w
    # Use a Stokes-like approach or just zero
    # Actually let's just use Newton with good settings
    
    # 8. Solve with Newton
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 50
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    # Use MUMPS if available
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 9. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
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
    u_values = np.full((points.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(n_newton),
            "nonlinear_iterations": [int(n_newton)],
        }
    }