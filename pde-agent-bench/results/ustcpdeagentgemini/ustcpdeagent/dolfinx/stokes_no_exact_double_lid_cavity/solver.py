import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Mesh resolution
    mesh_res = 128
    msh = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    nu = 0.3
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    bcs = []
    fdim = msh.topology.dim - 1
    
    # Top (y1) u = [1.0, 0.0]
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.full(x.shape[1], 1.0), np.zeros(x.shape[1])]))
    facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
    bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))
    
    # Right (x1) u = [0.0, -0.8]
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.full(x.shape[1], -0.8)]))
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs_right = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_right)
    bcs.append(fem.dirichletbc(u_right, dofs_right, W.sub(0)))
    
    # Left (x0) u = [0.0, 0.0]
    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
    facets_left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_left)
    bcs.append(fem.dirichletbc(u_left, dofs_left, W.sub(0)))
    
    # Bottom (y0) u = [0.0, 0.0]
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
    facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)
    bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))
    
    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
        
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample on grid
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
            
    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }

