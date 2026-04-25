import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    mesh_res = 128
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    nu = 0.2
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    bcs = []
    fdim = msh.topology.dim - 1
    
    # Boundary Conditions
    # y = 1 (Top)
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))
    
    # y = 0 (Bottom)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))
    
    # x = 0 (Left)
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_left, dofs_left, W.sub(0)))
    
    # x = 1 (Right)
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs_right = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    u_right = fem.Function(V)
    u_right.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_right, dofs_right, W.sub(0)))
    
    # Pin pressure
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
        
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    
    w_h = problem.solve()
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    
    # Evaluate
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    u_values = np.zeros((pts.shape[0], 2))
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals[:, :2]
        
    vel_mag = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": vel_mag, "solver_info": solver_info}

if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}}}
    res = solve(case)
    print("Max vel:", np.max(res["u"]))
