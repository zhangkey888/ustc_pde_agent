import numpy as np
from solver import solve

print("Testing edge cases...")
print("=" * 50)

# Test 1: Empty case_spec
print("\nTest 1: Empty case_spec")
try:
    result = solve({})
    print(f"  Success! Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"  Time: {result['solver_info']['wall_time_sec']:.2f}s")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: case_spec with only pde but no time
print("\nTest 2: case_spec with pde but no time")
try:
    result = solve({"pde": {}})
    print(f"  Success! Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"  Time: {result['solver_info']['wall_time_sec']:.2f}s")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: case_spec with partial time parameters
print("\nTest 3: case_spec with partial time parameters")
try:
    result = solve({"pde": {"time": {"t_end": 0.05}}})  # Only t_end
    print(f"  Success! Mesh resolution: {result['solver_info']['mesh_resolution']}")
    print(f"  dt used: {result['solver_info']['dt']}")
    print(f"  n_steps: {result['solver_info']['n_steps']}")
    print(f"  Time: {result['solver_info']['wall_time_sec']:.2f}s")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Different t_end
print("\nTest 4: t_end = 0.01")
try:
    result = solve({"pde": {"time": {"t_end": 0.01, "dt": 0.001}}})
    print(f"  Success! Mesh resolution: {result['solver_info']['mesh_resolution']}")
    
    # Compute error
    u_grid = result["u"]
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.exp(-0.01) * np.sin(4*np.pi*X) * np.sin(4*np.pi*Y)
    l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"  L2 error: {l2_error:.2e}")
    
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 50)
print("All edge case tests completed.")
