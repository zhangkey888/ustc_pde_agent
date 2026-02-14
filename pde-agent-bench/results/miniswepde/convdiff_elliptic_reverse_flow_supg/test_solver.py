import numpy as np
import time
from solver import solve

# Test case
case_spec = {
    "pde": {
        "type": "convection-diffusion",
        "epsilon": 0.02,
        "beta": [-8.0, 4.0]
    }
}

# Time the solve
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

print(f"Solve time: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Solver type: {result['solver_info']['ksp_type']}")
print(f"Preconditioner: {result['solver_info']['pc_type']}")
print(f"Total iterations: {result['solver_info']['iterations']}")

# Compute error against exact solution
u_grid = result['u']
nx, ny = u_grid.shape

# Exact solution on the same grid
def exact_solution(x, y):
    return np.exp(x) * np.sin(np.pi * y)

x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
u_exact = exact_solution(X, Y)

# Compute L2 error
error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f"L2 error: {error:.6e}")
print(f"Max error: {max_error:.6e}")

# Check against requirements
accuracy_req = 4.66e-04
time_req = 2.310
print(f"\nRequirements:")
print(f"  Accuracy: error ≤ {accuracy_req:.2e}")
print(f"  Time: ≤ {time_req:.3f}s")
print(f"\nStatus:")
print(f"  Accuracy: {'PASS' if error <= accuracy_req else 'FAIL'} (error = {error:.2e})")
print(f"  Time: {'PASS' if (end_time - start_time) <= time_req else 'FAIL'} (time = {end_time - start_time:.3f}s)")
