import numpy as np
import sys
sys.path.insert(0, '.')
from solver import solve

# Test 1: Normal case with time info
print("Test 1: Normal case with time info")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    },
    "coefficients": {
        "kappa": 1.0
    }
}
result1 = solve(case_spec1)
print(f"  Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"  dt: {result1['solver_info']['dt']}")
print(f"  n_steps: {result1['solver_info']['n_steps']}")
print(f"  ksp_type: {result1['solver_info']['ksp_type']}")
print(f"  pc_type: {result1['solver_info']['pc_type']}")
print(f"  iterations: {result1['solver_info']['iterations']}")
print(f"  Has wall_time_sec: {'wall_time_sec' in result1['solver_info']}")

# Test 2: Missing time key (should use defaults)
print("\nTest 2: Missing time key")
case_spec2 = {
    "coefficients": {
        "kappa": 1.0
    }
}
result2 = solve(case_spec2)
print(f"  dt: {result2['solver_info']['dt']}")
print(f"  n_steps: {result2['solver_info']['n_steps']}")

# Test 3: Partial time info
print("\nTest 3: Partial time info (only t_end)")
case_spec3 = {
    "pde": {
        "time": {
            "t_end": 0.05
        }
    },
    "coefficients": {
        "kappa": 1.0
    }
}
result3 = solve(case_spec3)
print(f"  dt: {result3['solver_info']['dt']}")
print(f"  n_steps: {result3['solver_info']['n_steps']}")

# Check accuracy
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(result1['u'] - u_exact)
max_error = np.max(error)
print(f"\nMax error for test 1: {max_error:.2e}")
print(f"Accuracy requirement: {max_error <= 1.42e-03}")

