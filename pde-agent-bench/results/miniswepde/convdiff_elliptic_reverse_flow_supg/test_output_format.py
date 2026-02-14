import numpy as np
from solver import solve

case_spec = {"pde": {"epsilon": 0.02, "beta": [-8.0, 4.0]}}
result = solve(case_spec)

print("Checking output format...")
print(f"1. 'u' key exists: {'u' in result}")
print(f"2. 'u' is numpy array: {type(result['u']).__name__}")
print(f"3. 'u' shape is (50, 50): {result['u'].shape == (50, 50)}")

print(f"\n4. 'solver_info' key exists: {'solver_info' in result}")
solver_info = result['solver_info']
required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
print("5. Required keys in solver_info:")
for key in required_keys:
    exists = key in solver_info
    value = solver_info.get(key, "MISSING")
    print(f"   - {key}: {exists} (value: {value})")

print(f"\n6. mesh_resolution is int: {isinstance(solver_info['mesh_resolution'], int)}")
print(f"7. element_degree is int: {isinstance(solver_info['element_degree'], int)}")
print(f"8. ksp_type is str: {isinstance(solver_info['ksp_type'], str)}")
print(f"9. pc_type is str: {isinstance(solver_info['pc_type'], str)}")
print(f"10. rtol is float: {isinstance(solver_info['rtol'], float)}")
print(f"11. iterations is int: {isinstance(solver_info['iterations'], int)}")

print("\nAll format checks passed!" if all([
    'u' in result,
    'solver_info' in result,
    result['u'].shape == (50, 50),
    all(key in solver_info for key in required_keys)
]) else "\nSome format checks failed!")
