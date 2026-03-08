import numpy as np
import time
from solver import solve

# Define the case specification
case_spec = {
    'pde': {
        'type': 'poisson',
        'coefficients': {
            'kappa': {'type': 'expr', 'expr': '1 + 0.3*sin(8*pi*x)*sin(8*pi*y)'}
        }
    },
    'domain': {'type': 'square', 'bounds': [[0,1], [0,1]]}
}

# Run solver with timing
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

wall_time = end_time - start_time

# Compute L2 error on the 50x50 grid
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y, indexing='ij')
u_exact = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
u_computed = result['u']

# L2 error (discrete approximation on the grid)
dx = 1/49  # spacing
dy = 1/49
error_sq = np.sum((u_computed - u_exact)**2) * dx * dy
l2_error = np.sqrt(error_sq)

print(f"Wall time: {wall_time:.6f} seconds")
print(f"L2 error (discrete): {l2_error:.6e}")
print(f"Max error: {np.abs(u_computed - u_exact).max():.6e}")
print(f"\nSolver info:")
for key, value in result['solver_info'].items():
    print(f"  {key}: {value}")

# Check constraints
accuracy_constraint = 1.28e-03
time_constraint = 2.569

print(f"\nConstraints:")
print(f"  Accuracy ≤ {accuracy_constraint}: {'PASS' if l2_error <= accuracy_constraint else 'FAIL'} (error = {l2_error:.6e})")
print(f"  Time ≤ {time_constraint}s: {'PASS' if wall_time <= time_constraint else 'FAIL'} (time = {wall_time:.6f}s)")
print(f"\nOverall: {'PASS' if l2_error <= accuracy_constraint and wall_time <= time_constraint else 'FAIL'}")
