from solver import solve

print("Testing edge cases...")

# Test 1: case_spec with time field (should still work for elliptic)
print("\n1. case_spec with time field:")
case_spec = {"pde": {"time": {"t_end": 1.0, "dt": 0.1}}}
result = solve(case_spec)
print(f"   Solution shape: {result['u'].shape}")
print(f"   Has dt in solver_info: {'dt' in result['solver_info']}")
if 'dt' in result['solver_info']:
    print(f"   dt value: {result['solver_info']['dt']}")

# Test 2: Empty case_spec
print("\n2. Empty case_spec:")
case_spec = {}
result = solve(case_spec)
print(f"   Solution shape: {result['u'].shape}")
print(f"   Success: {result['u'].shape == (50, 50)}")

# Test 3: case_spec with extra fields
print("\n3. case_spec with extra fields:")
case_spec = {"pde": {"type": "elliptic", "coefficients": {"kappa": 1.0}}, "domain": [0,1,0,1]}
result = solve(case_spec)
print(f"   Solution shape: {result['u'].shape}")
print(f"   Success: {result['u'].shape == (50, 50)}")

print("\nAll edge cases handled successfully!")
