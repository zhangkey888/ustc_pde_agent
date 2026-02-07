import time
import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "type": "elliptic",
        "time": None
    }
}

# Test current configuration
print("Testing current configuration (mesh_resolution=64, element_degree=2):")
start = time.time()
result = solve(case_spec)
elapsed = time.time() - start

u_grid = result["u"]
solver_info = result["solver_info"]

# Compute accuracy
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
max_error = np.max(np.abs(u_grid - u_exact))

print(f"  Time: {elapsed:.4f} seconds")
print(f"  Max error: {max_error:.6e}")
print(f"  Iterations: {solver_info['iterations']}")
print(f"  Pass accuracy: {max_error <= 5.81e-04}")

# Test with lower resolution
print("\nTesting with mesh_resolution=32, element_degree=2:")
# Modify solver temporarily - we'll create a modified version
import solver as original_solver

def solve_lower_res(case_spec):
    # Copy of solve function with lower resolution
    mesh_resolution = 32
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import ufl
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    ksp = problem.solver
    its = ksp.getIterationNumber()
    
    # Point evaluation (simplified - same as original)
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).astype(np.float64)
    
    from dolfinx import geometry
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
    
    local_u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        local_u_values[eval_map] = vals.flatten()
    
    # Gather to rank 0
    indices_array = np.array(eval_map, dtype=np.int32)
    indices_list = comm.gather(indices_array, root=0)
    values_array = local_u_values[~np.isnan(local_u_values)]
    values_list = comm.gather(values_array, root=0)
    
    if rank == 0:
        u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
        for proc_idx, (indices, values) in enumerate(zip(indices_list, values_list)):
            if indices is not None and values is not None:
                u_values[indices] = values
        
        if np.any(np.isnan(u_values)):
            nan_mask = np.isnan(u_values)
            if np.any(nan_mask):
                from scipy import spatial
                known_points = np.where(~nan_mask)[0]
                unknown_points = np.where(nan_mask)[0]
                if len(known_points) > 0 and len(unknown_points) > 0:
                    known_coords = points[:2, known_points].T
                    unknown_coords = points[:2, unknown_points].T
                    tree = spatial.KDTree(known_coords)
                    distances, indices = tree.query(unknown_coords)
                    u_values[unknown_points] = u_values[known_points[indices]]
        
        u_grid = u_values.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny), dtype=np.float64)
    
    u_grid = comm.bcast(u_grid, root=0)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": its
    }
    
    return {"u": u_grid, "solver_info": solver_info}

start = time.time()
result_lower = solve_lower_res(case_spec)
elapsed_lower = time.time() - start

u_grid_lower = result_lower["u"]
max_error_lower = np.max(np.abs(u_grid_lower - u_exact))

print(f"  Time: {elapsed_lower:.4f} seconds")
print(f"  Max error: {max_error_lower:.6e}")
print(f"  Iterations: {result_lower['solver_info']['iterations']}")
print(f"  Pass accuracy: {max_error_lower <= 5.81e-04}")

# Test with degree 1
print("\nTesting with mesh_resolution=64, element_degree=1:")

def solve_degree1(case_spec):
    mesh_resolution = 64
    element_degree = 1
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8
    
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import ufl
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    f_expr = 2.0 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 1000
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    ksp = problem.solver
    its = ksp.getIterationNumber()
    
    # Simplified point evaluation for testing
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Simple evaluation at grid points (not parallel-safe for this test)
    if rank == 0:
        points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).astype(np.float64)
        from dolfinx import geometry
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        
        points_on_proc = []
        cells_on_proc = []
        
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points.T[i])
                cells_on_proc.append(links[0])
        
        u_values = np.full((points.shape[1],), np.nan, dtype=np.float64)
        if len(points_on_proc) > 0:
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            # Map back - simplified for testing
            u_values[:len(vals)] = vals.flatten()
        
        u_grid = u_values.reshape((nx, ny))
    else:
        u_grid = np.zeros((nx, ny), dtype=np.float64)
    
    u_grid = comm.bcast(u_grid, root=0)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": its
    }
    
    return {"u": u_grid, "solver_info": solver_info}

start = time.time()
result_deg1 = solve_degree1(case_spec)
elapsed_deg1 = time.time() - start

u_grid_deg1 = result_deg1["u"]
max_error_deg1 = np.max(np.abs(u_grid_deg1 - u_exact))

print(f"  Time: {elapsed_deg1:.4f} seconds")
print(f"  Max error: {max_error_deg1:.6e}")
print(f"  Iterations: {result_deg1['solver_info']['iterations']}")
print(f"  Pass accuracy: {max_error_deg1 <= 5.81e-04}")

print("\nSummary:")
print(f"Original (64, degree 2): {elapsed:.4f}s, error={max_error:.2e}")
print(f"Lower res (32, degree 2): {elapsed_lower:.4f}s, error={max_error_lower:.2e}")
print(f"Lower degree (64, degree 1): {elapsed_deg1:.4f}s, error={max_error_deg1:.2e}")
