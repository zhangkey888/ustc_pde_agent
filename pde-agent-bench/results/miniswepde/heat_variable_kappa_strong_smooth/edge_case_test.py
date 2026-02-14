import numpy as np
from solver import solve

print("Test 1: t_end not divisible by dt")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.03,  # 0.1/0.03 = 3.333, not integer
            "scheme": "backward_euler"
        }
    }
}
try:
    result1 = solve(case_spec1)
    print(f"  Success! n_steps = {result1['solver_info']['n_steps']}")
    print(f"  Actual steps computed: {int(0.1/0.03)}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nTest 2: Very small dt (many steps)")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.001,  # 100 steps
            "scheme": "backward_euler"
        }
    }
}
try:
    result2 = solve(case_spec2)
    print(f"  Success! n_steps = {result2['solver_info']['n_steps']}")
    print(f"  Iterations: {result2['solver_info']['iterations']}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nTest 3: Empty case_spec")
case_spec3 = {}
try:
    result3 = solve(case_spec3)
    print(f"  Success! Used defaults")
    print(f"  dt: {result3['solver_info']['dt']}, t_end: 0.1 (default)")
except Exception as e:
    print(f"  Failed: {e}")

print("\nTest 4: Missing time key")
case_spec4 = {"pde": {}}
try:
    result4 = solve(case_spec4)
    print(f"  Success! Used defaults")
    print(f"  dt: {result4['solver_info']['dt']}, n_steps: {result4['solver_info']['n_steps']}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nAll edge case tests completed!")
