import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # 1. Mesh
    mesh_res = 64
    elem_deg = 2
    domain = mesh.create_rectangle(
        comm, 
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
        [mesh_res, mesh_res], 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # 2. Function Space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # 3. Exact solution and parameters (UFL)
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.5 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1])
    u_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Automatic source term generation
    f = -ufl.div(kappa * ufl.grad(u_ex))
    
    # 4. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x_pts):
        return np.full(x_pts.shape[1], True) # All boundaries
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 5. Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-8},
        petsc_options_prefix="pdebench_"
    )
    u_sol = problem.solve()
    
    # 7. Sampling on output grid
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
            
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    # Gather solver info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": problem.solver.getIterationNumber()
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Simple test
    spec = {
        "output": {
            "grid": {
                "nx": 50, "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(spec)
    print("Max u:", np.nanmax(res["u"]))
    print("Info:", res["solver_info"])
