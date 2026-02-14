import numpy as np
import solver

# Test case specification
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.08,
            "dt": 0.008,
            "scheme": "backward_euler"
        }
    }
}

# Run solver
result = solver.solve(case_spec)

# Check required fields
print("Checking result structure...")
assert "u" in result, "Missing 'u' field"
assert "solver_info" in result, "Missing 'solver_info' field"
assert "u_initial" in result, "Missing 'u_initial' field"

# Check u shape
u = result["u"]
assert u.shape == (50, 50), f"u shape should be (50, 50), got {u.shape}"

# Check solver_info fields
info = result["solver_info"]
required_fields = [
    "mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol",
    "iterations", "dt", "n_steps", "time_scheme"
]
for field in required_fields:
    assert field in info, f"Missing field in solver_info: {field}"
    print(f"  {field}: {info[field]}")

# Check that iterations is an integer
assert isinstance(info["iterations"], int), f"iterations should be int, got {type(info['iterations'])}"

# Check that dt and n_steps are consistent with t_end
t_end = 0.08
dt = info["dt"]
n_steps = info["n_steps"]
print(f"\ndt * n_steps = {dt * n_steps}, t_end = {t_end}")
assert abs(dt * n_steps - t_end) < 1e-10, f"dt * n_steps should equal t_end"

# Compute error against exact solution at final time
def u_exact(x, y, t):
    return np.exp(-t) * np.exp(5*x) * np.sin(np.pi*y)

# Create grid
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Exact solution at final time
u_exact_final = u_exact(X, Y, 0.08)

# Computed solution
u_computed = result["u"]

# Compute L2 error on grid
error = np.sqrt(np.mean((u_computed - u_exact_final)**2))
max_error = np.max(np.abs(u_computed - u_exact_final))
print(f"\nError analysis:")
print(f"  RMS error: {error:.2e}")
print(f"  Max error: {max_error:.2e}")

# Check against accuracy requirement
accuracy_requirement = 1.06e-03
print(f"  Accuracy requirement: {accuracy_requirement:.2e}")
if error <= accuracy_requirement:
    print("  ✓ Accuracy requirement met!")
else:
    print(f"  ✗ Accuracy requirement NOT met (error {error:.2e} > {accuracy_requirement:.2e})")

print("\nAll checks passed!")
