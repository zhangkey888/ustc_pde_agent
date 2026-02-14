import numpy as np
import time
from solver import solve

# Load reference solution
ref_data = np.load('oracle_output/reference.npz')
x_ref = ref_data['x']  # shape (55,)
y_ref = ref_data['y']  # shape (55,)
u_star = ref_data['u_star']  # shape (55, 55)

case_spec = {
    "epsilon": 0.01,
    "beta": [-12.0, 6.0]
}

# Time the solve
start = time.time()
result = solve(case_spec)
end = time.time()

u_computed = result['u']
solver_info = result['solver_info']

# Compute L2 error on the grid
# The grid is uniform 55x55, same as reference
dx = 1.0 / 54.0  # spacing
dy = 1.0 / 54.0
error_grid = u_computed - u_star
l2_error = np.sqrt(np.sum(error_grid**2) * dx * dy)
max_error = np.max(np.abs(error_grid))

print(f"Time taken: {end - start:.4f} s")
print(f"Mesh resolution used: {solver_info['mesh_resolution']}")
print(f"Element degree: {solver_info['element_degree']}")
print(f"Solver iterations: {solver_info['iterations']}")
print(f"L2 error: {l2_error:.6e}")
print(f"Max error: {max_error:.6e}")

# Check against pass/fail criteria
accuracy_required = 1.07e-04
time_required = 2.887
if l2_error <= accuracy_required and (end - start) <= time_required:
    print("PASS: Meets both accuracy and time constraints.")
else:
    print("FAIL: Does not meet constraints.")
    if l2_error > accuracy_required:
        print(f"  Accuracy: {l2_error:.2e} > {accuracy_required:.2e}")
    if (end - start) > time_required:
        print(f"  Time: {end - start:.3f} > {time_required:.3f}")
