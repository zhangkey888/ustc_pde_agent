import time
import numpy as np
from solver import solve

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 0.5},
        "source": "sin(3*pi*x)*sin(2*pi*y)"
    }
}

# Warm-up run
result = solve(case_spec)

# Timed run
start_time = time.time()
result = solve(case_spec)
end_time = time.time()

elapsed = end_time - start_time
print(f"Solve time: {elapsed:.3f} seconds")
print(f"Time constraint: ≤ 2.980 seconds")
print(f"Meets time constraint: {elapsed <= 2.980}")

# Verify accuracy again
ref_data = np.load('oracle_output/reference.npz')
u_star = ref_data['u_star']
u_computed = result['u']
x_ref = ref_data['x']
y_ref = ref_data['y']

dx = x_ref[1] - x_ref[0]
dy = y_ref[1] - y_ref[0]
error = np.sqrt(np.sum((u_computed - u_star)**2) * dx * dy)
print(f"\nL2 error: {error:.6e}")
print(f"Accuracy constraint: ≤ 2.24e-02")
print(f"Meets accuracy constraint: {error <= 2.24e-02}")
