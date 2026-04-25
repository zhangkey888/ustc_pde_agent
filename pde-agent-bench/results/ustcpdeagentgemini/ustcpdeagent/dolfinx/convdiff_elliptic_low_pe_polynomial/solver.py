import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the steady convection-diffusion equation with manufactured solution.
    """
    # Extract output grid parameters
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    xmin, xmax, ymin, ymax = case_spec["output"]["grid"]["bbox"]
    
    # Discretization parameters
    mesh_res = 32
    element_degree = 2
    
    # 1. Mesh and Function Space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 2. Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # 3. Parameters
    eps = 0.3
    beta = ufl.as_vector([0.5, 0.3])
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Source term (derived from manufactured solution u = x*(1-x)*y*(1-y))
    lap_u = -2.0*x[1]*(1.0-x[1]) - 2.0*x[0]*(1.0-x[0])
    grad_u_x = (1.0-2.0*x[0])*x[1]*(1.0-x[1])
    grad_u_y = x[0]*(1.0-x[0])*(1.0-2.0*x[1])
    f = -eps * lap_u + 0.5 * grad_u_x + 0.3 * grad_u_y
    
    # 5. Variational form
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary Conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Solver setup and execution
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # 8. Sample onto uniform grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
    
    # 9. Formulate solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
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
    # Test script execution
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    print("Test solve completed successfully.")
    print("Max u:", np.nanmax(result["u"]))
