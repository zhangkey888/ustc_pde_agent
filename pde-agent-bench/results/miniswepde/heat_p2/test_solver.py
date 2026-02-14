import numpy as np
from solver import solve
import time

# Define exact solution for verification
def exact_solution(x, y, t):
    return np.exp(-t) * (x**2 + y**2)

# Create case specification
case_spec = {
    'pde': {
        'time': {
            't_end': 0.06,
            'dt': 0.01,
            'scheme': 'backward_euler'
        }
    },
    'coefficients': {
        'kappa': 1.0
    },
    'domain': {
        'bounds': [[0.0, 0.0], [1.0, 1.0]]
    }
}

# Run solver
start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Wall time: {end - start:.3f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Time steps: {result['solver_info']['n_steps']}")
print(f"Total linear iterations: {result['solver_info']['iterations']}")

# Compute error on 50x50 grid
u_grid = result['u']
nx, ny = u_grid.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution at final time t=0.06
u_exact = exact_solution(X, Y, 0.06)

# Compute L2 error (approximate)
error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f"RMS error: {error:.2e}")
print(f"Max error: {max_error:.2e}")

# Check pass/fail criteria
accuracy_required = 1.08e-03
time_required = 8.729
if error <= accuracy_required and (end - start) <= time_required:
    print("PASS: Meets both accuracy and time constraints")
else:
    print("FAIL: Does not meet constraints")
    if error > accuracy_required:
        print(f"  Accuracy: {error:.2e} > {accuracy_required:.2e}")
    if (end - start) > time_required:
        print(f"  Time: {end - start:.3f} > {time_required:.3f}")
