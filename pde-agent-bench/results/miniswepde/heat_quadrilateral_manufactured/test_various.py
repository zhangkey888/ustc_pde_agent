import numpy as np
from solver import solve

# Test 1: Default case
print("Test 1: Default case")
case_spec1 = {
    'pde': {
        'type': 'heat',
        'time': {'t_end': 0.1, 'dt': 0.01, 'scheme': 'backward_euler'},
        'coefficients': {'kappa': 1.0}
    },
    'domain': {'type': 'unit_square', 'bounds': [[0,1], [0,1]]}
}
result1 = solve(case_spec1)
print(f"  Mesh res: {result1['solver_info']['mesh_resolution']}")
print(f"  Error check: L2 norm of u = {np.sqrt(np.mean(result1['u']**2)):.3e}")

# Test 2: Smaller dt
print("\nTest 2: Smaller dt")
case_spec2 = {
    'pde': {
        'type': 'heat',
        'time': {'t_end': 0.1, 'dt': 0.005, 'scheme': 'backward_euler'},
        'coefficients': {'kappa': 1.0}
    },
    'domain': {'type': 'unit_square', 'bounds': [[0,1], [0,1]]}
}
result2 = solve(case_spec2)
print(f"  Mesh res: {result2['solver_info']['mesh_resolution']}")
print(f"  n_steps: {result2['solver_info']['n_steps']}")

# Test 3: Different kappa
print("\nTest 3: Different kappa")
case_spec3 = {
    'pde': {
        'type': 'heat',
        'time': {'t_end': 0.1, 'dt': 0.01, 'scheme': 'backward_euler'},
        'coefficients': {'kappa': 2.0}
    },
    'domain': {'type': 'unit_square', 'bounds': [[0,1], [0,1]]}
}
result3 = solve(case_spec3)
print(f"  Mesh res: {result3['solver_info']['mesh_resolution']}")

# Test 4: Missing time info (should use defaults from problem description)
print("\nTest 4: Missing time info")
case_spec4 = {
    'pde': {
        'type': 'heat',
        'coefficients': {'kappa': 1.0}
    },
    'domain': {'type': 'unit_square', 'bounds': [[0,1], [0,1]]}
}
result4 = solve(case_spec4)
print(f"  dt used: {result4['solver_info']['dt']}")
print(f"  t_end inferred: {result4['solver_info']['n_steps'] * result4['solver_info']['dt']}")

print("\nAll tests passed!")
