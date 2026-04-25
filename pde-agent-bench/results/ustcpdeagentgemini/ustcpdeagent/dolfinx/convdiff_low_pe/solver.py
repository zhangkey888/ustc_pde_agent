import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # 1. Parse specifications
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 64
    degree = 2
    
    # 2. Setup Mesh & Function Space
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm, 
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])], 
        [mesh_res, mesh_res], 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Define parameters and manufactured source term
    eps = 0.2
    beta = ufl.as_vector([1.0, 0.5])
    
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    f = eps * 2.0 * ufl.pi**2 * u_exact + \
        beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + \
        beta[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
        
    # 4. Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 5. Variational Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Solve
    ksp_type = "preonly"
    pc_type = "lu"
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # 7. Evaluate on target grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
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
        
    u_grid = u_values.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
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
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    t0 = time.time()
    res = solve(case_spec)
    t1 = time.time()
    
    u = res["u"]
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((u - u_exact)**2))
    print(f"L2 Error: {error:.4e}")
    print(f"Wall time: {t1-t0:.4f}s")
