import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx_mesh, ny_mesh = 64, 64
    msh = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 1.0
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    bcs = []
    fdim = msh.topology.dim - 1
    
    # x0
    facets_x0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_x0, dofs_x0, W.sub(0)))
    
    # y0
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_y0 = fem.Function(V)
    u_y0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y0, dofs_y0, W.sub(0)))
    
    # y1
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    u_y1 = fem.Function(V)
    u_y1.interpolate(lambda x: np.vstack((np.full(x.shape[1], 0.5), np.zeros(x.shape[1]))))
    bcs.append(fem.dirichletbc(u_y1, dofs_y1, W.sub(0)))
    
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    
    out_grid = case_spec["output"]["grid"]
    nx, ny = out_grid["nx"], out_grid["ny"]
    bbox = out_grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack((XX.ravel(), YY.ravel(), np.zeros(nx*ny)))
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_vals = np.zeros((nx*ny, gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape((ny, nx))
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }
