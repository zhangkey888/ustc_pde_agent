import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh setup
    mesh_res = 64
    degree = 1
    msh = mesh.create_rectangle(comm, [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
                                [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(msh, ("Lagrange", degree))
    
    # Time params
    t0 = 0.0
    t_end = 0.25
    dt = 0.005
    n_steps = int(np.round((t_end - t0) / dt))
    
    eps_val = case_spec.get("pde", {}).get("epsilon", 0.01)
    
    # Initial Condition
    u_n = fem.Function(V)
    x = ufl.SpatialCoordinate(msh)
    u0_expr = 0.1 * ufl.exp(-50 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    # Save initial state for probing
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Boundary Conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Source term
    f_expr = 5 * ufl.exp(-180 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Variational problem
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    v = ufl.TestFunction(V)
    
    # Backward Euler, Allen-Cahn reaction R(u) = u^3 - u
    F = (u - u_n) / dt * v * ufl.dx \
      + eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
      + (u**3 - u) * v * ufl.dx \
      - f * v * ufl.dx
      
    J = ufl.derivative(F, u)
    
    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="ac_")
    
    solver = petsc.PETSc.SNES().create(comm)
    solver.setOptionsPrefix("ac_")
    solver.setType("newtonls")
    solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=20)
    
    ksp = solver.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    
    # Time loop
    nonlinear_iters = []
    
    for i in range(n_steps):
        # We can just use problem.solve() which embeds SNES in dolfinx 0.10.0,
        # but let's do it via direct solver creation or just calling solve()
        pass
    
    # Actually, let's use the simpler petsc.NonlinearProblem approach recommended:
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    nlp = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix="ac_", petsc_options=petsc_options)
    
    for i in range(n_steps):
        num_its, converged = nlp.solve()
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
        nonlinear_iters.append(num_its)
        
    # Interpolation onto grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i_pt, pt in enumerate(pts):
        if len(colliding.links(i_pt)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding.links(i_pt)[0])
            eval_map.append(i_pt)
            
    u_vals = np.full((nx * ny,), np.nan)
    u_init_vals = np.full((nx * ny,), np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells, dtype=np.int32)
        u_vals[eval_map] = u.eval(pts_arr, cells_arr).flatten()
        u_init_vals[eval_map] = u_initial_func.eval(pts_arr, cells_arr).flatten()
        
    u_grid = u_vals.reshape(ny, nx)
    u_init_grid = u_init_vals.reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-6,
        "iterations": sum(nonlinear_iters), # approx
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }
