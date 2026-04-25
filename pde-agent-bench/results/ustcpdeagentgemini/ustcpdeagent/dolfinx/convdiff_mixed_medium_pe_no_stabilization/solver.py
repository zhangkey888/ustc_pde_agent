import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract parameters
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Parameters
    epsilon = fem.Constant(domain, PETSc.ScalarType(0.02))
    beta = fem.Constant(domain, PETSc.ScalarType((6.0, 2.0)))
    
    # Exact solution and RHS
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_expr = -epsilon * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form
    F = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
      + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx \
      - ufl.inner(f_expr, v) * ufl.dx
      
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * vnorm)
    
    residual = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    F += tau * ufl.inner(residual, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a, L = ufl.system(F)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    
    # Evaluation
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
            
    u_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": u_grid, "solver_info": solver_info}

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
    start = time.time()
    res = solve(case_spec)
    end = time.time()
    print(f"Time: {end-start:.3f} s")
    
    # Check error
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(2*np.pi*XX)*np.sin(2*np.pi*YY)
    error = np.sqrt(np.mean((res["u"] - u_exact)**2))
    print(f"RMSE: {error:.2e}")
