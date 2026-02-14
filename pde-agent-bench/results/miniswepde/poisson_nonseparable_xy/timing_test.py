import time
import numpy as np
from mpi4py import MPI
from solver import solve

comm = MPI.COMM_WORLD
rank = comm.rank

# Warm up
if rank == 0:
    print("Warming up...")

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0}
    }
}

# Time the solve
if rank == 0:
    print("Starting timed solve...")
    start_time = time.time()

result = solve(case_spec)

if rank == 0:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal solve time: {elapsed:.6f} seconds")
    print(f"Time constraint: ≤ 1.931 seconds")
    
    if elapsed <= 1.931:
        print(f"PASS: Time {elapsed:.6f}s ≤ 1.931s")
    else:
        print(f"FAIL: Time {elapsed:.6f}s > 1.931s")
    
    # Also compute L2 error more accurately
    from dolfinx import mesh, fem
    import ufl
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    
    # Recreate the solution to compute error
    N = result['solver_info']['mesh_resolution']
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(np.pi * x[0] * x[1])
    g_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(g_expr)
    
    # We don't have the actual function object, but we can approximate error
    # from the 50x50 grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact_grid = np.sin(np.pi * X * Y)
    
    # Compute L2 error on the grid (approximate)
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    error_sq = np.sum((result['u'] - u_exact_grid)**2) * dx * dy
    error_l2 = np.sqrt(error_sq)
    
    print(f"\nApproximate L2 error on 50x50 grid: {error_l2:.6e}")
    print(f"Accuracy requirement: ≤ 1.08e-04")
    
    if error_l2 <= 1.08e-04:
        print(f"PASS: Error {error_l2:.6e} ≤ 1.08e-04")
    else:
        print(f"FAIL: Error {error_l2:.6e} > 1.08e-04")
