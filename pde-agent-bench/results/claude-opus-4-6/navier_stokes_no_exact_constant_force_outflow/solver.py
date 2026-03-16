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
    nu_val = 0.3
    mesh_resolution = 64
    degree_u = 2
    degree_p = 1
    newton_rtol = 1e-8
    newton_atol = 1e-10
    newton_max_it = 50
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
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
    
    # Define solution function and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term and viscosity
    f = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0)))
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Nonlinear residual: u·∇u - ν∇²u + ∇p = f, ∇·u = 0
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary conditions
    # Identify boundary facets
    # We need: no-slip on walls (top, bottom, left), outflow on right
    # For "outflow" on right boundary (x=1), we apply natural BC (do-nothing)
    # For walls: u = g. Since no exact solution, g = 0 on walls typically
    # But let's check case_spec for boundary conditions
    
    # Left boundary: x = 0
    def left(x):
        return np.isclose(x[0], 0.0)
    
    # Top boundary: y = 1
    def top(x):
        return np.isclose(x[1], 1.0)
    
    # Bottom boundary: y = 0
    def bottom(x):
        return np.isclose(x[1], 0.0)
    
    # Right boundary: x = 1 (outflow - natural BC, no Dirichlet)
    
    bcs = []
    
    # Zero velocity on left, top, bottom
    u_zero = fem.Function(V)
    u_zero.interpolate(lambda x: np.zeros((2, x.shape[1])))
    
    for marker_func in [left, top, bottom]:
        facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bc = fem.dirichletbc(u_zero, dofs, W.sub(0))
        bcs.append(bc)
    
    # Pin pressure at one point to remove nullspace (since we have outflow BC)
    # Actually with do-nothing outflow, pressure is determined. But let's be safe.
    # With outflow (natural BC), the pressure is determined, so no pinning needed.
    
    # Initial guess: solve Stokes first for better convergence
    # First do a Stokes solve (drop convection term)
    w_stokes = fem.Function(W)
    (u_s, p_s) = ufl.split(w_stokes)
    (v_s, q_s) = ufl.TestFunctions(W)
    
    F_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - ufl.inner(p_s, ufl.div(v_s)) * ufl.dx
        - ufl.inner(f, v_s) * ufl.dx
        + ufl.inner(ufl.div(u_s), q_s) * ufl.dx
    )
    
    # Solve Stokes as nonlinear (it's actually linear, but this is simpler)
    problem_stokes = petsc.NonlinearProblem(F_stokes, w_stokes, bcs=bcs)
    solver_stokes = nls.petsc.NewtonSolver(comm, problem_stokes)
    solver_stokes.convergence_criterion = "incremental"
    solver_stokes.rtol = 1e-8
    solver_stokes.atol = 1e-10
    solver_stokes.max_it = 5
    
    ksp_stokes = solver_stokes.krylov_solver
    ksp_stokes.setType(PETSc.KSP.Type.GMRES)
    pc_stokes = ksp_stokes.getPC()
    pc_stokes.setType(PETSc.PC.Type.LU)
    
    n_stokes, converged_stokes = solver_stokes.solve(w_stokes)
    w_stokes.x.scatter_forward()
    
    # Use Stokes solution as initial guess
    w.x.array[:] = w_stokes.x.array[:]
    
    # Now solve full Navier-Stokes
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.atol = newton_atol
    solver.max_it = newton_max_it
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-8)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
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
    
    vel_magnitude = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": newton_rtol,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }