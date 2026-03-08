import numpy as np
import time
from solver import solve

def compute_residual(u_grid, k=15.0):
    """Compute residual of Helmholtz equation on 50x50 grid using finite differences."""
    nx, ny = u_grid.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # Source term f = 10*exp(-80*((x-0.35)**2 + (y-0.55)**2))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 10.0 * np.exp(-80.0 * ((X - 0.35)**2 + (Y - 0.55)**2))
    
    # Compute Laplacian using central finite differences
    laplacian = np.zeros_like(u_grid)
    
    # Interior points
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            laplacian[i, j] = (u_grid[i+1, j] - 2*u_grid[i, j] + u_grid[i-1, j]) / (dx**2) + \
                              (u_grid[i, j+1] - 2*u_grid[i, j] + u_grid[i, j-1]) / (dy**2)
    
    # Boundary points (use one-sided differences or set to 0 since BCs are satisfied)
    # For error estimation, we can use the interior points only
    residual = -laplacian - k**2 * u_grid - f
    
    # Compute L2 norm of residual (interior only)
    interior_mask = np.zeros_like(u_grid, dtype=bool)
    interior_mask[1:-1, 1:-1] = True
    residual_interior = residual[interior_mask]
    
    l2_residual = np.sqrt(np.mean(residual_interior**2)) * dx * dy
    max_residual = np.max(np.abs(residual_interior))
    
    return l2_residual, max_residual, residual

# Run solver
case_spec = {
    'pde': {'k': 15.0},
    'domain': {'bounds': [[0, 0], [1, 1]]}
}

start = time.time()
result = solve(case_spec)
elapsed = time.time() - start

u_grid = result['u']
l2_res, max_res, residual = compute_residual(u_grid, k=15.0)

print(f"Time: {elapsed:.3f} s")
print(f"Mesh: {result['solver_info']['mesh_resolution']}")
print(f"Degree: {result['solver_info']['element_degree']}")
print(f"L2 residual (interior): {l2_res:.6e}")
print(f"Max residual (interior): {max_res:.6e}")
print(f"Accuracy requirement: error ≤ 1.63e-01")
print(f"Time requirement: ≤ 14.614 s")
print(f"Time check: {'PASS' if elapsed <= 14.614 else 'FAIL'}")
print(f"Residual check: {'PASS' if l2_res <= 1.63e-01 else 'FAIL'} (using residual as proxy)")
