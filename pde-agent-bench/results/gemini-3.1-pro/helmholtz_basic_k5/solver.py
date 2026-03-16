import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nx, ny = 100, 100
    degree = 2
    k = 5.0
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    # -Laplacian(u) = 2*pi^2 * u
    # f = -Laplacian(u) - k^2 * u = (2*pi^2 - k^2) * u
    f_expr = (2 * ufl.pi**2 - k**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Get iterations
    solver = problem.solver
    iterations = solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    x_lin = np.linspace(0, 1, 50)
    y_lin = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())])
    
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
        
    # Handle parallel evaluation safely (assuming single process for standard evaluation, but safe for MPI)
    u_values_local = np.where(np.isnan(u_values), 0.0, u_values)
    u_values_global = np.zeros_like(u_values_local)
    comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape((50, 50))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }