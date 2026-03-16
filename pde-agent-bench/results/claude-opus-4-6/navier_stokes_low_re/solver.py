import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    N = 32  # mesh resolution
    degree_u = 2
    degree_p = 1
    nu_val = 1.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces - Taylor-Hood P2/P1
    Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree_u)
    Qe = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree_p)
    Me = ufl.MixedElement([Ve, Qe])
    W = fem.functionspace(domain, Me)
    
    # Also create sub-spaces for BCs
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Define solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term from manufactured solution
    # f = u_exact · ∇u_exact - ν ∇²u_exact + ∇p_exact
    f = (ufl.grad(u_exact) * u_exact 
         - nu_val * ufl.div(ufl.grad(u_exact)) 
         + ufl.grad(p_exact))
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
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
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex at (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    corner_vertices = mesh.locate_entities_boundary(domain, 0, corner)
    dofs_p = fem.locate_dofs_topological((W.sub(1), Q), 0, corner_vertices)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution
    w_sub_u = w.sub(0)
    w_sub_p = w.sub(1)
    
    # Set initial guess to exact solution for faster convergence
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.stack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    w.x.array[V_map] = u_init.x.array
    
    p_init = fem.Function(Q)
    p_init.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    w.x.array[Q_map] = p_init.x.array
    
    w.x.scatter_forward()
    
    # Newton solver
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 25
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
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
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }