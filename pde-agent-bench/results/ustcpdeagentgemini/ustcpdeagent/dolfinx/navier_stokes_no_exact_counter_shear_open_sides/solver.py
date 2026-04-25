import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_info = case_spec.get("output", {}).get("grid", {})
    nx = grid_info.get("nx", 64)
    ny = grid_info.get("ny", 64)
    bbox = grid_info.get("bbox", [0, 1, 0, 1])
    
    # Physical parameters
    nu = 0.2
    
    # Mesh definition
    mesh_res = 64
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Function space: Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    bcs = []
    
    # Top BC: u = [0.8, 0.0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 0.8), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))
    
    # Bottom BC: u = [-0.8, 0.0]
    bot_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bot_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bot_facets)
    u_bot = fem.Function(V)
    u_bot.interpolate(lambda x: np.vstack([np.full(x.shape[1], -0.8), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_bot, bot_dofs, W.sub(0)))
    
    # Pin pressure at origin to fix the nullspace
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
        
    # Weak form formulation
    def eps(u_): return ufl.sym(ufl.grad(u_))
    def sigma(u_, p_): return 2.0 * nu * eps(u_) - p_ * ufl.Identity(gdim)
    
    f = ufl.as_vector((0.0, 0.0))
    
    # Nonlinear residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Jacobian
    J = ufl.derivative(F, w)
    
    # Solve Stokes first to provide a good initial guess
    F_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    problem_stokes = petsc.NonlinearProblem(
        F_stokes, w, bcs=bcs, J=ufl.derivative(F_stokes, w), 
        petsc_options_prefix="stokes_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem_stokes.solve()
    
    # Solve Navier-Stokes
    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    
    problem_ns = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J, 
        petsc_options_prefix="ns_", 
        petsc_options=petsc_options
    )
    
    # Run solver
    num_newton_its, _ = problem_ns.solve()
    w.x.scatter_forward()
    
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Interpolate output onto the grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(msh, gdim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_vals = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": num_newton_its, # Approximate linear solves
        "nonlinear_iterations": [num_newton_its]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
