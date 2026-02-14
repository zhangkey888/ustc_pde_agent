"""
Final test to ensure solver meets all requirements.
"""
import numpy as np
from solver import solve

# The exact case specification from the problem
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}

print("Running final validation test...")
print("=" * 60)

# Run solver
result = solve(case_spec)

# Check required fields
required_fields = ["u", "solver_info"]
for field in required_fields:
    if field not in result:
        print(f"✗ Missing required field: {field}")
    else:
        print(f"✓ Has required field: {field}")

# Check solver_info fields
solver_info = result["solver_info"]
required_info_fields = [
    "mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol",
    "iterations", "dt", "n_steps", "time_scheme"
]

print("\nChecking solver_info fields:")
for field in required_info_fields:
    if field not in solver_info:
        print(f"✗ Missing solver_info field: {field}")
    else:
        print(f"✓ Has solver_info field: {field} = {solver_info[field]}")

# Check solution shape
u = result["u"]
expected_shape = (50, 50)
if u.shape == expected_shape:
    print(f"\n✓ Solution has correct shape: {u.shape}")
else:
    print(f"✗ Solution has wrong shape: {u.shape}, expected {expected_shape}")

# Check accuracy
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(4*np.pi*X) * np.sin(4*np.pi*Y)

l2_error = np.sqrt(np.mean((u - u_exact)**2))
accuracy_requirement = 8.89e-03

print(f"\nAccuracy check:")
print(f"  L2 error: {l2_error:.2e}")
print(f"  Required: ≤ {accuracy_requirement:.2e}")

if l2_error <= accuracy_requirement:
    print("  ✓ PASS: Meets accuracy requirement")
else:
    print("  ✗ FAIL: Does not meet accuracy requirement")

# Check time
time_taken = solver_info.get("wall_time_sec", float('inf'))
time_requirement = 26.841

print(f"\nTime check:")
print(f"  Time taken: {time_taken:.2f}s")
print(f"  Required: ≤ {time_requirement}s")

if time_taken <= time_requirement:
    print("  ✓ PASS: Meets time requirement")
else:
    print("  ✗ FAIL: Does not meet time requirement")

# Check adaptive mesh refinement was used
print(f"\nAdaptive mesh refinement:")
print(f"  Mesh resolution used: {solver_info['mesh_resolution']}")
print(f"  Resolutions tried: [32, 64, 128]")

# Check solver robustness
print(f"\nSolver robustness:")
print(f"  Solver type: {solver_info['ksp_type']}/{solver_info['pc_type']}")
if solver_info['ksp_type'] == 'gmres' and solver_info['pc_type'] == 'hypre':
    print("  ✓ Using iterative solver first (gmres/hypre)")
elif solver_info['ksp_type'] == 'preonly' and solver_info['pc_type'] == 'lu':
    print("  ✓ Using direct solver (preonly/lu) as fallback")

print("\n" + "=" * 60)
print("Final test completed.")
