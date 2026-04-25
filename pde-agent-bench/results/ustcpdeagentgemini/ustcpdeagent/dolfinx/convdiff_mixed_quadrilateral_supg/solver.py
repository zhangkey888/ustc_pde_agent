import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # Extract grid parameters
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Problem parameters
    eps = 0.01
    beta_val = [14.0, 6.0]
    
    # Mesh resolution
    nx = 48
    ny = 48
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    beta = ufl.as_vector(beta_val)
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG parameter tau
    # For high Peclet number: tau = h / (2 * |beta|)
    tau = h / (2.0 * beta_norm)
    
    # Residual
    def residual(u_func):
        return -eps * ufl.div(ufl.grad(u_func)) + ufl.dot(beta, ufl.grad(u_func)) - f
    
    # Add SUPG terms
    a_supg = a + tau * ufl.dot(beta, ufl.grad(v)) * (-eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L_supg = L + tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx
    
    # Boundary Conditions (u=0 on boundary)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="cd_"
    )
    u_h = problem.solve()
    
    # Evaluate on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
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
            
    u_out = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_out = u_out.reshape((ny_out, nx_out))
    
    # Fill solver info
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_out,
        "solver_info": solver_info
    }
