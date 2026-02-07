import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation: -∇·(κ ∇u) = f in Ω, u = g on ∂Ω
    Manufactured solution: u = sin(pi*x)*sin(pi*y)
    κ = 1.0
    
    Returns:
        dict with keys:
        - "u": solution array shape (50, 50) sampled on uniform grid
        - "solver_info": dict with solver parameters and performance
    """
    # Optimized parameters for accuracy and speed
    mesh_resolution = 128      # Higher resolution for better accuracy
    element_degree = 2         # Quadratic elements for better accuracy
    ksp_type = "cg"            # Conjugate gradient for symmetric positive definite
    pc_type = "hypre"          # Algebraic multigrid preconditioner  
    rtol = 1e-10               # Tight tolerance for high accuracy
    
    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create mesh
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define manufactured solution using UFL
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # κ = 1.0 (given)
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Compute source term f from manufactured solution
    # f = -∇·(κ ∇u) = κ * (2*pi²) * sin(pi*x)*sin(pi*y) since κ=1
    # ∇²(sin(pi*x)*sin(pi*y)) = -2*pi² * sin(pi*x)*sin(pi*y)
    # So f = κ * 2*pi² * sin(pi*x)*sin(pi*y)
    f_expr = kappa * (2.0 * ufl.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Boundary condition: u = g = sin(pi*x)*sin(pi*y) on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        # Mark all boundaries
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    # Locate boundary facets
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    
    # Locate DOFs on boundary
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create boundary condition function
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: κ * ∇u·∇v = f * v
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Create forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Create solution function
    u_sol = fem.Function(V)
    
    # Assemble matrix with boundary conditions
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = petsc.create_vector(L_form.function_spaces)
    
    # Assemble RHS vector
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    
    # Apply boundary conditions to RHS
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    # Set up linear solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
    solver.setOperators(A)
    
    # Solve the linear system
    u_sol_petsc = u_sol.x.petsc_vec
    solver.solve(b, u_sol_petsc)
    u_sol.x.scatter_forward()
    
    # Get iteration count
    iterations = solver.getIterationNumber()
    
    # Sample solution on 50x50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Create points array: shape (3, nx*ny) with z=0 for 2D
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Build bounding box tree for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
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
    
    # Evaluate solution at points
    local_u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        local_u_values[eval_map] = vals.flatten()
    
    # Gather results to rank 0
    indices_array = np.array(eval_map, dtype=np.int32)
    indices_list = comm.gather(indices_array, root=0)
    
    values_array = local_u_values[~np.isnan(local_u_values)]
    values_list = comm.gather(values_array, root=0)
    
    # Rank 0 assembles the complete grid
    if rank == 0:
        u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
        
        # Combine results from all processes
        for proc_idx, (indices, values) in enumerate(zip(indices_list, values_list)):
            if indices is not None and values is not None:
                u_values[indices] = values
        
        # Check for any remaining NaN values
        nan_mask = np.isnan(u_values)
        if np.any(nan_mask):
            # Simple fallback: use nearest known value
            import scipy.spatial
            
            known_indices = np.where(~nan_mask)[0]
            unknown_indices = np.where(nan_mask)[0]
            
            if len(known_indices) > 0 and len(unknown_indices) > 0:
                # Get coordinates
                known_coords = points[:2, known_indices].T
                unknown_coords = points[:2, unknown_indices].T
                
                # Find nearest known point for each unknown point
                tree = scipy.spatial.KDTree(known_coords)
                distances, nearest_idx = tree.query(unknown_coords)
                u_values[unknown_indices] = u_values[known_indices[nearest_idx]]
        
        # Reshape to (nx, ny)
        u_grid = u_values.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny), dtype=np.float64)
    
    # Broadcast the grid to all processes
    u_grid = comm.bcast(u_grid, root=0)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver
    case_spec = {
        "pde": {
            "type": "elliptic",
            "time": None  # No time dependence for Poisson
        }
    }
    result = solve(case_spec)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Poisson equation solver completed successfully")
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solver info: {result['solver_info']}")
        
        # Quick accuracy check
        nx, ny = 50, 50
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        max_error = np.max(np.abs(result['u'] - u_exact))
        print(f"Max error: {max_error:.6e}")
        print(f"Accuracy requirement (≤ 5.81e-04): {max_error <= 5.81e-04}")
