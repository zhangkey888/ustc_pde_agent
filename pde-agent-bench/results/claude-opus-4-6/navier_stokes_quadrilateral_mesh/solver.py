import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = 0.1
    N = 32  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    # Create quadrilateral mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Define mixed element (Taylor-Hood on quads)
    vel_el = element("Lagrange", domain.basix_cell(), degree_u, shape=(2,))
    pres_el = element("Lagrange", domain.basix_cell(), degree_p)
    mel = mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mel)
    
    # Extract sub-spaces for BCs
    V_sub, _ = W.sub(0).collapse()
    
    # Define solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    pi = ufl.pi
    u_exact = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                              -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    p_exact = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Compute source term from manufactured solution
    # u·∇u - ν ∇²u + ∇p = f
    # f = u_exact·∇u_exact - ν ∇²u_exact + ∇p_exact
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Residual form
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions: u = u_exact on all boundaries
    u_bc_func = fem.Function(V_sub)
    u_bc_func.interpolate(lambda X: np.array([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
        -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex near (0,0)
    Q_sub, _ = W.sub(1).collapse()
    
    def corner(X):
        return np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), corner)
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution for faster convergence
    w.sub(0).interpolate(lambda X: np.array([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
        -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))
    
    # Solve nonlinear problem
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
    
    u_magnitude = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]
    
    u_grid = u_magnitude.reshape((nx_eval, ny_eval))
    
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