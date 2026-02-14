import numpy as np
from solver import solve

# Load reference solution
ref_data = np.load('oracle_output/reference.npz')
u_star = ref_data['u_star']
x_ref = ref_data['x']
y_ref = ref_data['y']

# Run solver
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 0.5},
        "source": "sin(3*pi*x)*sin(2*pi*y)"
    }
}

result = solve(case_spec)
u_computed = result['u']

# Check grid matches
assert u_computed.shape == u_star.shape, f"Shape mismatch: {u_computed.shape} vs {u_star.shape}"

# Compute L2 error
dx = x_ref[1] - x_ref[0]
dy = y_ref[1] - y_ref[0]
error = np.sqrt(np.sum((u_computed - u_star)**2) * dx * dy)
max_error = np.max(np.abs(u_computed - u_star))
mean_error = np.mean(np.abs(u_computed - u_star))

print(f"L2 error: {error:.6e}")
print(f"Max absolute error: {max_error:.6e}")
print(f"Mean absolute error: {mean_error:.6e}")
print(f"Accuracy requirement: ≤ 2.24e-02")
print(f"L2 error meets requirement: {error <= 2.24e-02}")
print(f"Max error meets requirement: {max_error <= 2.24e-02}")

# Also print solver info
print(f"\nSolver info: {result['solver_info']}")
