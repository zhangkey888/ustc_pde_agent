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
    
    # Manufactured solution
    # u = [-40*(y-0.5)*exp(-20*((x-0.5)**2 + (y-0.5)**2)),
    #       40*(x-0.5)*exp(-20*((x-0.5)**2 + (y-0.5)**2))]
    # p = 0
    
    # 2. Create mesh - use sufficient resolution for accuracy
    N = 80
    degree_u = 2
    degree_p = 1
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Create mixed element
    vel_elem = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    pres_elem = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    mixed_elem = ufl.MixedElement([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    # 4. Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Exact solution in UFL
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    exp_term = ufl.exp(-20.0 * r2)
    u_exact_0 = -40.0 * (x[1] - 0.5) * exp_term
    u_exact_1 = 40.0 * (x[0] - 0.5) * exp_term
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.Constant(domain, PETSc.ScalarType(0.0))
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    # Since p_exact = 0, ∇p_exact = 0
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.array([
        -40.0 * (x[1] - 0.5) * np.exp(-20.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
        40.0 * (x[0] - 0.5) * np.exp(-20.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure pin (fix one DOF to remove nullspace)
    # Find a point near corner (0,0)
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    # Pin pressure at a vertex
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), corner_point)
    p_bc_func = fem.Function(Q)
    p_bc_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 6. Initial guess - interpolate exact solution as initial guess for fast convergence
    w_sub0 = w.sub(0)
    
    # Set initial guess to exact solution for velocity
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.array([
        -40.0 * (x[1] - 0.5) * np.exp(-20.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)),
        40.0 * (x[0] - 0.5) * np.exp(-20.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    ]))
    
    # We need to interpolate into the mixed space
    # Use sub-space interpolation
    w.sub(0).interpolate(u_init)
    w.x.scatter_forward()
    
    # 7. Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(domain.comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 30
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # 8. Extract velocity on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    u_sol = w.sub(0).collapse()
    
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
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