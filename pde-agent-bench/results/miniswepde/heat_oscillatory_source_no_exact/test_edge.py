import solver
import numpy as np

# Test 1: Empty case_spec
print("Test 1: Empty case_spec")
result1 = solver.solve({})
print(f"Mesh resolution used: {result1['solver_info']['mesh_resolution']}")
print(f"dt used: {result1['solver_info']['dt']}")
print(f"t_end implied: {result1['solver_info']['n_steps'] * result1['solver_info']['dt']}")

# Test 2: Partial case_spec
print("\nTest 2: Partial case_spec")
case_spec2 = {
    "pde": {
        "time": {
            "dt": 0.01  # only dt, no t_end
        }
    }
}
result2 = solver.solve(case_spec2)
print(f"Mesh resolution used: {result2['solver_info']['mesh_resolution']}")
print(f"dt used: {result2['solver_info']['dt']}")
print(f"n_steps: {result2['solver_info']['n_steps']}")

# Test 3: Different kappa
print("\nTest 3: Different kappa")
case_spec3 = {
    "coefficients": {
        "kappa": 1.5
    }
}
result3 = solver.solve(case_spec3)
print(f"Kappa in solver info not stored, but solution computed.")

print("\nAll edge tests completed.")
