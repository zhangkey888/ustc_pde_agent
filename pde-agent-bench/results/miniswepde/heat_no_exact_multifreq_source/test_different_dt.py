import numpy as np
from solver import solve

# Test with different dt that doesn't divide t_end exactly
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.03,  # 0.12 / 0.03 = 4 exactly, but let's test
            "scheme": "backward_euler"
        }
    }
}

print("Testing with dt=0.03...")
result = solve(case_spec)
info = result["solver_info"]
print(f"dt used: {info['dt']}")
print(f"n_steps: {info['n_steps']}")
print(f"dt * n_steps = {info['dt'] * info['n_steps']:.6f}")
assert abs(info['dt'] * info['n_steps'] - 0.12) < 1e-10, "Time stepping doesn't reach t_end"
print("Test passed!")

# Test with very small dt
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.001,
            "scheme": "backward_euler"
        }
    }
}

print("\nTesting with dt=0.001...")
result2 = solve(case_spec2)
info2 = result2["solver_info"]
print(f"dt used: {info2['dt']}")
print(f"n_steps: {info2['n_steps']}")
print(f"dt * n_steps = {info2['dt'] * info2['n_steps']:.6f}")
assert abs(info2['dt'] * info2['n_steps'] - 0.12) < 1e-10, "Time stepping doesn't reach t_end"
print("Test passed!")
