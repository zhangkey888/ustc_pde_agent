import numpy as np
import time
from solver import solve

print("=== Final Comprehensive Test ===\n")

# Test 1: Full case_spec
print("Test 1: Full case_spec")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
start = time.time()
result1 = solve(case_spec1)
time1 = time.time() - start
print(f"Time: {time1:.3f}s")
print(f"Mesh resolution: {result1['solver_info']['mesh_resolution']}")
print(f"Element degree: {result1['solver_info']['element_degree']}")

# Check accuracy
u_grid = result1['u']
nx, ny = 50, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))
print(f"Max error: {max_error:.6e}")
print(f"L2 error: {l2_error:.6e}")
print(f"Accuracy OK: {l2_error <= 1.80e-03}")
print(f"Time OK: {time1 <= 16.635}")

# Test 2: Empty case_spec (should use defaults)
print("\nTest 2: Empty case_spec")
case_spec2 = {}
start = time.time()
result2 = solve(case_spec2)
time2 = time.time() - start
print(f"Time: {time2:.3f}s")
print(f"dt: {result2['solver_info']['dt']}")
print(f"n_steps: {result2['solver_info']['n_steps']}")
print(f"t_end implied: {result2['solver_info']['dt'] * result2['solver_info']['n_steps']}")

# Test 3: Different dt
print("\nTest 3: Different dt (0.005)")
case_spec3 = {
    "pde": {
        "time": {
            "dt": 0.005
        }
    }
}
start = time.time()
result3 = solve(case_spec3)
time3 = time.time() - start
print(f"Time: {time3:.3f}s")
print(f"dt: {result3['solver_info']['dt']}")
print(f"n_steps: {result3['solver_info']['n_steps']}")

print("\n=== All tests completed ===")
