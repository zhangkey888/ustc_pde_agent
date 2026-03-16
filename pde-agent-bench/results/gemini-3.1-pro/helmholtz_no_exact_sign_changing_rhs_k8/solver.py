import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 128, 128
    degree = 2
    k = 8.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions: u = 0 on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x_spatial = ufl.SpatialCoordinate(domain)
    f = ufl.cos(4 * ufl.pi * x_spatial[0]) * ufl.sin(3 * ufl.pi * x_spatial[1])
    
    # Weak form: -∇²u - k²u = f
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve using a direct solver (LU) since the problem is indefinite
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
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
            
    u_values_local = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values_local[eval_map] = vals.flatten()
        
    # Handle parallel evaluation safely
    u_values_clean = np.where(np.isnan(u_values_local), 0.0, u_values_local)
    has_data = (~np.isnan(u_values_local)).astype(np.int32)
    
    u_values_global = np.zeros_like(u_values_clean)
    counts = np.zeros_like(has_data)
    
    comm.Allreduce(u_values_clean, u_values_global, op=MPI.SUM)
    comm.Allreduce(has_data, counts, op=MPI.SUM)
    
    valid = counts > 0
    u_values_global[valid] /= counts[valid]
    
    u_grid = u_values_global.reshape((50, 50))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }