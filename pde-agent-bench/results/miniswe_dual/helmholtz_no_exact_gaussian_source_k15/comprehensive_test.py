import numpy as np
import time
from solver import solve

def run_test(k_value, test_name):
    print(f"\n=== {test_name} (k={k_value}) ===")
    case_spec = {
        'pde': {'k': k_value},
        'domain': {'bounds': [[0, 0], [1, 1]]}
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    u_grid = result['u']
    info = result['solver_info']
    
    # Compute residual
    nx, ny = u_grid.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 10.0 * np.exp(-80.0 * ((X - 0.35)**2 + (Y - 0.55)**2))
    
    laplacian = np.zeros_like(u_grid)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            laplacian[i, j] = (u_grid[i+1, j] - 2*u_grid[i, j] + u_grid[i-1, j]) / (dx**2) + \
                              (u_grid[i, j+1] - 2*u_grid[i, j] + u_grid[i, j-1]) / (dy**2)
    
    residual = -laplacian - k_value**2 * u_grid - f
    interior_mask = np.zeros_like(u_grid, dtype=bool)
    interior_mask[1:-1, 1:-1] = True
    residual_interior = residual[interior_mask]
    
    l2_residual = np.sqrt(np.mean(residual_interior**2)) * dx * dy
    max_residual = np.max(np.abs(residual_interior))
    
    print(f"Time: {elapsed:.3f}s (limit: 14.614s) - {'PASS' if elapsed <= 14.614 else 'FAIL'}")
    print(f"Mesh: {info['mesh_resolution']}, Degree: {info['element_degree']}")
    print(f"L2 residual: {l2_residual:.6e}")
    print(f"Max residual: {max_residual:.6e}")
    print(f"Accuracy check (max residual ≤ 1.63e-01): {'PASS' if max_residual <= 1.63e-01 else 'FAIL'}")
    
    return elapsed <= 14.614 and max_residual <= 1.63e-01

# Test various k values
k_values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
all_pass = True

for k in k_values:
    passed = run_test(k, f"Test k={k}")
    all_pass = all_pass and passed

print(f"\n=== SUMMARY ===")
print(f"All tests passed: {all_pass}")
print(f"Main test case (k=15) should meet: error ≤ 1.63e-01, time ≤ 14.614s")
