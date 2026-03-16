import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = 0.01
    N = 64  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces (Taylor-Hood P2/P1)
    V_el = ("Lagrange", degree_u, (domain.geometry.dim,))
    Q_el = ("Lagrange", degree_p)
    
    V = fem.functionspace(domain, V_el)
    Q = fem.functionspace(domain, Q_el)
    
    # Mixed function space
    from dolfinx.fem import Function
    from basix.ufl import mixed_element, element
    
    vel_elem = element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    pres_elem = element("Lagrange", domain.basix_cell(), degree_p)
    mixed_el = mixed_element([vel_elem, pres_elem])
    
    W = fem.functionspace(domain, mixed_el)
    
    # Current solution
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    pi = ufl.pi
    u_exact_0 = 0.2 * pi * ufl.cos(pi * x[1]) * ufl.sin(2 * pi * x[0])
    u_exact_1 = -0.4 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Compute source term from manufactured solution
    # f = u·∇u - ν ∇²u + ∇p
    # Since p=0, ∇p = 0
    # f = (u_exact · ∇)u_exact - ν Δu_exact
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.dot(grad_u_exact, u_exact)
    laplacian_exact = ufl.div(ufl.grad(u_exact))
    
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = convection_exact - nu_c * laplacian_exact
    
    # Residual form
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions - all boundaries
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    
    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        0.2 * np.pi * np.cos(np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -0.4 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Initial guess: interpolate exact solution to help convergence
    W0 = W.sub(0)
    w.sub(0).interpolate(lambda x: np.stack([
        0.2 * np.pi * np.cos(np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -0.4 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 30
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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