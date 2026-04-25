import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm, 
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle
    )
    print(f"Mesh created in {time.time()-t0:.2f}s")
    
    el = basix_element("Lagrange", domain.topology.cell_name(), degree)
    W = fem.functionspace(domain, basix_mixed_element([el, el]))
    
    V_u, _ = W.sub(0).collapse()
    V_v, _ = W.sub(1).collapse()
    
    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    f = 4.0 * ufl.pi**4 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    a = (ufl.inner(ufl.grad(u), ufl.grad(phi)) - ufl.inner(v, phi) +
         ufl.inner(ufl.grad(v), ufl.grad(psi))) * ufl.dx
    L = ufl.inner(f, psi) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    
    u0 = fem.Function(V_u)
    u0.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u0, fem.locate_dofs_topological((W.sub(0), V_u), fdim, boundary_facets), W.sub(0))
    
    v0 = fem.Function(V_v)
    v0.x.array[:] = 0.0
    bc_v = fem.dirichletbc(v0, fem.locate_dofs_topological((W.sub(1), V_v), fdim, boundary_facets), W.sub(1))
    
    bcs = [bc_u, bc_v]
    
    print(f"Setup done in {time.time()-t0:.2f}s")
    
    t_solve = time.time()
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="biharmonic_"
    )
    w_sol = problem.solve()
    print(f"Solve done in {time.time()-t_solve:.2f}s")
    
    u_sol = w_sol.sub(0).collapse()
    
    t_eval = time.time()
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.flatten(), YY.flatten(), np.zeros_like(XX.flatten())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    print(f"Eval done in {time.time()-t_eval:.2f}s")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
