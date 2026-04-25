import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element, mixed_element
import ufl
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Mesh resolution
    mesh_res = 64
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood elements P2/P1
    vel_el = element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    nu = 1.0
    f = fem.Constant(msh, np.zeros(gdim))
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    fdim = msh.topology.dim - 1
    
    bcs = []
    
    # y0 (bottom, y=0): u = [0, 0]
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_y0 = fem.Function(V)
    u_y0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y0, dofs_y0, W.sub(0)))
    
    # y1 (top, y=1): u = [0, 0]
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    u_y1 = fem.Function(V)
    u_y1.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y1, dofs_y1, W.sub(0)))
    
    # x0 (left, x=0): u = [4*y*(1-y), 0]
    facets_x0 = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack((4 * x[1] * (1 - x[1]), np.zeros_like(x[1]))))
    bcs.append(fem.dirichletbc(u_x0, dofs_x0, W.sub(0)))
    
    # Pressure pin: fix p(0,0) = 0
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Solve
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_sol, p_sol = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    
    # Interpolation onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_vals = np.zeros((nx_out * ny_out, gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": magnitude, "solver_info": solver_info}

if __name__ == "__main__":
    spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    res = solve(spec)
    print("Max magnitude:", np.max(res["u"]))
