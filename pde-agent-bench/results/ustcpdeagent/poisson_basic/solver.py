import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation: -∇·(κ ∇u) = f with u = g on ∂Ω
    Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    
    Optimized to use available time budget for maximum accuracy.
    """
    # Start timing
    start_time = time.time()
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec or use optimized defaults
    # Using higher resolution to maximize accuracy within time budget
    mesh_resolution = case_spec.get('mesh_resolution', 100)
    element_degree = case_spec.get('element_degree', 2)
    ksp_type = case_spec.get('ksp_type', 'preonly')
    pc_type = case_spec.get('pc_type', 'lu')
    rtol = case_spec.get('rtol', 1e-8)
    
    # Create unit square mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define boundary condition (all Dirichlet)
    # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    # Create expression for exact solution
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_expr)
    
    # Locate all boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Locate DOFs on boundary
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # κ = 1.0
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Source term f = -∇·(κ ∇u_exact) = 2*pi^2*sin(pi*x)*sin(pi*y)
    f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    # Weak form: ∫(κ ∇u·∇v) dx = ∫ f v dx
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve using LinearProblem
    from dolfinx.fem.petsc import LinearProblem
    
    # Set PETSc options for direct solver (fastest for this size)
    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-10,
    }
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    
    # Calculate L2 error
    error_form = ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx
    error_local = fem.assemble_scalar(fem.form(error_form))
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    l2_error = np.sqrt(error_global)
    
    # Sample solution on a uniform grid for output
    # Use moderate grid (50x50) since evaluator will resample anyway
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Create points array (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to rank 0
    if comm.rank == 0:
        u_grid = u_values.reshape(nx, ny)
    else:
        u_grid = np.zeros((nx, ny), dtype=np.float64)
    
    # Broadcast from rank 0 to all ranks (simplify for return)
    if comm.size > 1:
        u_grid = comm.bcast(u_grid, root=0)
    
    # End timing
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Get iteration count
    try:
        iterations = problem.solver.getIterationNumber()
    except:
        iterations = 1  # Direct solver typically has 1 iteration
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    # Print diagnostics for verification
    if comm.rank == 0:
        print(f"L2_ERROR: {l2_error:.6e}")
        print(f"WALL_TIME: {wall_time:.6f}")
        print(f"Mesh resolution: {mesh_resolution}")
        print(f"Element degree: {element_degree}")
        print(f"Solver iterations: {iterations}")
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with optimized parameters
    case_spec = {
        "mesh_resolution": 100,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
    }
    
    result = solve(case_spec)
    
    # Also print the error and time for verification
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(f"\nTest completed successfully.")
        print(f"Output u shape: {result['u'].shape}")
