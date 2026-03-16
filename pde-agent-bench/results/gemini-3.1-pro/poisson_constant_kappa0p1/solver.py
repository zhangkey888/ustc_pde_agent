import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def probe_points(u_func, points_array, domain):
    """
    u_func: dolfinx.fem.Function
    points_array: shape (3, N) numpy array
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    # Build per-point mapping (avoid boolean mask mismatch)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points_array.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    kappa = PETSc.ScalarType(0.1)
    
    # Mesh
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # Function Space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
        )
        
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational Problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    # f = -div(kappa * grad(u)) = 0.2 * pi^2 * sin(pi*x) * sin(pi*y)
    f_expr = 0.2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="pdebench_"
    )
    
    u_sol = problem.solve()
    
    # Get iterations
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    x_coords = np.linspace(0, 1, nx_eval)
    y_coords = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.vstack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))
    
    u_eval = probe_points(u_sol, points, domain)
    u_grid = u_eval.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }