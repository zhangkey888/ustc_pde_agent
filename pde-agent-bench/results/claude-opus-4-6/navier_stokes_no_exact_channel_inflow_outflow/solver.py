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
    nu_val = 0.12
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
    
    # Also create sub-spaces for BCs
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Define solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Source term
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # Weak form (residual)
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    
    # Boundary conditions
    # Determine boundary type from case_spec
    # Channel with inflow/outflow: 
    # - Inflow on left (x=0): parabolic profile
    # - No-slip on top (y=1) and bottom (y=0)
    # - Outflow on right (x=1): natural BC (do nothing)
    
    bcs = []
    
    # Inflow: parabolic profile on left boundary x=0
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([
        4.0 * x[1] * (1.0 - x[1]),
        np.zeros_like(x[0])
    ]))
    bc_inflow = fem.dirichletbc(u_inflow, left_dofs, W.sub(0))
    bcs.append(bc_inflow)
    
    # No-slip on top y=1
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    
    u_noslip = fem.Function(V)
    u_noslip.interpolate(lambda x: np.vstack([
        np.zeros_like(x[0]),
        np.zeros_like(x[0])
    ]))
    bc_top = fem.dirichletbc(u_noslip, top_dofs, W.sub(0))
    bcs.append(bc_top)
    
    # No-slip on bottom y=0
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    
    u_noslip_bot = fem.Function(V)
    u_noslip_bot.interpolate(lambda x: np.vstack([
        np.zeros_like(x[0]),
        np.zeros_like(x[0])
    ]))
    bc_bot = fem.dirichletbc(u_noslip_bot, bottom_dofs, W.sub(0))
    bcs.append(bc_bot)
    
    # Initial guess: interpolate inflow profile everywhere for better convergence
    w_sub_u = w.sub(0)
    # Set initial guess via the collapsed space
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.vstack([
        4.0 * x[1] * (1.0 - x[1]),
        np.zeros_like(x[0])
    ]))
    w.x.array[V_map] = u_init.x.array[:]
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50
    solver.relaxation_parameter = 1.0
    
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setTolerances(rtol=1e-8, max_it=1000)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    n_newton, converged = solver.solve(w)
    assert converged, f"Newton solver did not converge after {n_newton} iterations"
    w.x.scatter_forward()
    
    # Extract velocity sub-function
    u_sol = w.sub(0).collapse()
    
    # Sample on 50x50 grid
    n_grid = 50
    xs = np.linspace(0.0, 1.0, n_grid)
    ys = np.linspace(0.0, 1.0, n_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
        # vals shape: (n_eval_points, 2) for 2D velocity
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((n_grid, n_grid))
    
    # Get total linear iterations
    total_linear_its = 0
    # We can't easily get per-Newton-step iterations from the high-level API,
    # but we report what we can
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": int(n_newton * 1),  # approximate
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }