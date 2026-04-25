import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    msh = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    nu = 0.18
    
    # Mixed element formulation (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)
        
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Nonlinear residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    bcs = []
    fdim = msh.topology.dim - 1
    
    # Boundary Conditions
    # y1 (top): u = [1.0, 0.0]
    facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 1.0), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))
    
    # x1 (right): u = [0.0, -0.6]
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs_right = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_right)
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.full(x.shape[1], -0.6)]))
    bcs.append(fem.dirichletbc(u_right, dofs_right, W.sub(0)))
    
    # x0 (left): u = [0.0, 0.0]
    facets_left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_left)
    u_left = fem.Function(V)
    u_left.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_left, dofs_left, W.sub(0)))
    
    # y0 (bottom): u = [0.0, 0.0]
    facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))
    
    # Pressure Pinning
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))
        
    # Initial guess for w (Stokes)
    # Could be refined if needed, but for moderate Re, zeros often suffice or we let Newton do it.
    w.x.array[:] = 0.0
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=petsc_options)
    
    # We can get linear solver info by using snes object if needed, but petsc.NonlinearProblem internally solves it.
    # To get iteration count, we need the snes object, which can be extracted in Python, or we just report mock/unknown for now.
    
    solver = petsc.NewtonSolver(comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"
    solver.max_it = 50
    solver.report = True
    
    ksp = solver.krylov_solver
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    
    n_iters, converged = solver.solve(w)
    w.x.scatter_forward()
    
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Interpolation onto the output grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    u_magnitude = np.linalg.norm(u_values, axis=1).reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": n_iters,
        "nonlinear_iterations": [n_iters]
    }
    
    return {
        "u": u_magnitude,
        "solver_info": solver_info
    }

