import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import dolfinx.fem.petsc
import ufl

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract grid info
    grid_info = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_info.get("nx", 128)
    ny_out = grid_info.get("ny", 128)
    bbox = grid_info.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Parameters
    epsilon = 0.02
    beta_val = [8.0, 3.0]
    
    # Mesh and Function Space
    mesh_res = 128
    degree = 1
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.exp(-250 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Galerkin terms
    F_galerkin = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
               + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx \
               - f * v * ufl.dx
               
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    vnorm = ufl.sqrt(ufl.inner(beta, beta))
    tau = h / (2.0 * vnorm)
    
    # Strong residual (for P1, laplacian is 0)
    residual = -eps_const * ufl.div(ufl.grad(u)) + ufl.inner(beta, ufl.grad(u)) - f
    F_supg = F_galerkin + ufl.inner(residual, tau * ufl.inner(beta, ufl.grad(v))) * ufl.dx
    
    a, L = ufl.lhs(F_supg), ufl.rhs(F_supg)
    
    # Solve
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    
    # Sample onto grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)
            
    u_grid = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        u_eval = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[eval_map] = u_eval.flatten()
        
    u_grid = u_grid.reshape((ny_out, nx_out))
    
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
    spec = {
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    res = solve(spec)
    print("Max u:", np.max(res["u"]))
    print("Done")
