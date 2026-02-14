import solver
import numpy as np

# Test case
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        }
    },
    "coefficients": {
        "kappa": 0.8
    }
}

result = solver.solve(case_spec)

# Check output structure
assert "u" in result
assert "u_initial" in result
assert "solver_info" in result

u = result["u"]
u0 = result["u_initial"]
info = result["solver_info"]

# Check shapes
assert u.shape == (50, 50)
assert u0.shape == (50, 50)

# Check solver_info required fields
required = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
            "iterations", "dt", "n_steps", "time_scheme"]
for field in required:
    assert field in info, f"Missing field: {field}"
    print(f"{field}: {info[field]}")

# Check types
assert isinstance(info["mesh_resolution"], int)
assert isinstance(info["element_degree"], int)
assert isinstance(info["iterations"], int)
assert isinstance(info["n_steps"], int)
assert isinstance(info["dt"], float)
assert isinstance(info["rtol"], float)

print("All tests passed!")
