import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Increased mesh resolution for better accuracy
    mesh_res = 128
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    gdim = domain.geometry.dim
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    
    nu = 0.8
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    bcs = []
    
    # x = 0: u = [2*y*(1-y), 2*y*(1-y)]
    facets_x0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack((2*x[1]*(1-x[1]), 2*x[1]*(1-x[1]))))
    bcs.append(fem.dirichletbc(u_x0, dofs_x0, W.sub(0)))
    
    # y = 0: u = [0, 0]
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_y0 = fem.Function(V)
    u_y0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y0, dofs_y0, W.sub(0)))
    
    # y = 1: u = [0, 0]
    facets_y1 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    u_y1 = fem.Function(V)
    u_y1.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y1, dofs_y1, W.sub(0)))
    
    # Solve
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_vals = np.full((pts.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape((ny_out, nx_out))
    
    # Fill any NaNs with 0.0 (though there shouldn't be any if bbox is inside domain)
    magnitude = np.nan_to_num(magnitude, nan=0.0)
    
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

