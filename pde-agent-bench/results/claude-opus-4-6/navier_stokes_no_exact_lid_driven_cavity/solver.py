import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, nls, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 0.08
    nx = ny = 64
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space (Taylor-Hood P2/P1)
    vel_elem = element("Lagrange", domain.basix_cell(), degree_u, shape=(2,))
    pres_elem = element("Lagrange", domain.basix_cell(), degree_p)
    mel = mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mel)
    
    # Also create individual spaces for BC interpolation
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Define boundary conditions
    # Lid-driven cavity: u=(1,0) on top, u=(0,0) on other walls
    
    # Top boundary (y=1): u = (1, 0)
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    # Bottom boundary (y=0)
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    # Left boundary (x=0)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    # Right boundary (x=1)
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    # No-slip walls (bottom, left, right)
    def noslip_boundary(x):
        return np.isclose(x[1], 0.0) | np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    
    # Create BC functions
    u_noslip = fem.Function(V)
    u_noslip.x.array[:] = 0.0
    
    u_lid = fem.Function(V)
    u_lid.interpolate(lambda x: np.vstack([np.ones_like(x[0]), np.zeros_like(x[0])]))
    
    # Locate facets and DOFs
    noslip_facets = mesh.locate_entities_boundary(domain, fdim, noslip_boundary)
    lid_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    
    noslip_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, noslip_facets)
    lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
    
    bc_noslip = fem.dirichletbc(u_noslip, noslip_dofs, W.sub(0))
    bc_lid = fem.dirichletbc(u_lid, lid_dofs, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a DOF near (0,0) for pressure
    pressure_point_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    # Use geometrical approach for pressure pin
    def corner_point(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), corner_point)
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_pressure = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
    
    bcs = [bc_noslip, bc_lid, bc_pressure]
    
    # Define variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # Residual form
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Set initial guess: zero (or could do Stokes first)
    w.x.array[:] = 0.0
    
    # Create nonlinear problem and Newton solver
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    # Configure KSP
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setGMRESRestart(100)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
    
    # Solve
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # Extract velocity and pressure
    u_sol = w.sub(0).collapse()
    p_sol = w.sub(1).collapse()
    
    # Evaluate velocity magnitude on 50x50 grid
    n_grid = 50
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_grid * n_grid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
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
    
    vel_mag = np.full(n_grid * n_grid, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((n_grid, n_grid))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-8,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }