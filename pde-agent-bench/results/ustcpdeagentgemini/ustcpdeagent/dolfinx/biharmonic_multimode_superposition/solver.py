import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx = grid_spec.get("nx", 50)
    ny = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])

    comm = MPI.COMM_WORLD
    
    mesh_resolution = 128
    element_degree = 2
    
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    f = 4.0 * pi**4 * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) + \
        84.5 * pi**4 * ufl.sin(2.0*pi*x[0]) * ufl.sin(3.0*pi*x[1])
        
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x_coord):
        return np.logical_or(np.logical_or(np.isclose(x_coord[0], 0.0), np.isclose(x_coord[0], 1.0)),
                             np.logical_or(np.isclose(x_coord[1], 0.0), np.isclose(x_coord[1], 1.0)))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc_val = fem.Function(V)
    bc_val.x.array[:] = 0.0
    bc = fem.dirichletbc(bc_val, boundary_dofs)
    
    # Step 1: Solve -Delta v = f  (where v = -Delta u)
    a_v = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
    L_v = ufl.inner(f, w) * ufl.dx
    
    problem_v = petsc.LinearProblem(a_v, L_v, bcs=[bc],
                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                    petsc_options_prefix="pdev_")
    v_sol = problem_v.solve()
    
    # Step 2: Solve -Delta u = v
    a_u = ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
    L_u = ufl.inner(v_sol, q) * ufl.dx
    
    problem_u = petsc.LinearProblem(a_u, L_u, bcs=[bc],
                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                    petsc_options_prefix="pdeu_")
    u_sol = problem_u.solve()
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree_u = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree_u, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros(len(pts))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 2
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(case_spec)
    print("Max u:", np.max(res["u"]))
