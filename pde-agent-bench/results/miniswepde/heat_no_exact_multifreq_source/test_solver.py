import numpy as np
from solver import solve

# Test case specification
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    }
}

print("Testing solver...")
result = solve(case_spec)

# Check outputs
assert "u" in result, "Missing 'u' in result"
assert "solver_info" in result, "Missing 'solver_info' in result"
assert "u_initial" in result, "Missing 'u_initial' in result"

u = result["u"]
info = result["solver_info"]
u0 = result["u_initial"]

print(f"Solution shape: {u.shape}")
print(f"Initial condition shape: {u0.shape}")
print(f"Solution min/max: {u.min():.6f}, {u.max():.6f}")
print(f"Initial min/max: {u0.min():.6f}, {u0.max():.6f}")

# Check solver info fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                   "iterations", "dt", "n_steps", "time_scheme"]
for field in required_fields:
    assert field in info, f"Missing field '{field}' in solver_info"
    print(f"{field}: {info[field]}")

# Check that mesh_resolution is one of the expected values
assert info["mesh_resolution"] in [32, 64, 128], f"Unexpected mesh_resolution: {info['mesh_resolution']}"

# Check that n_steps * dt is approximately t_end
t_end = 0.12
dt = info["dt"]
n_steps = info["n_steps"]
computed_t_end = n_steps * dt
assert abs(computed_t_end - t_end) < 1e-10, f"Time stepping mismatch: {n_steps} * {dt} = {computed_t_end} != {t_end}"

print("All tests passed!")
