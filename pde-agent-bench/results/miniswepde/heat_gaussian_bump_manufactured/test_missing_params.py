import time
import numpy as np

# Test 1: Complete case_spec
print("Test 1: Complete case_spec")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

from solver import solve

start_time = time.time()
result1 = solve(case_spec1)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.3f} seconds")
print("Mesh resolution:", result1["solver_info"]["mesh_resolution"])
print("dt used:", result1["solver_info"]["dt"])
print("n_steps:", result1["solver_info"]["n_steps"])
print()

# Test 2: Missing time parameters (should use defaults)
print("Test 2: Missing time parameters")
case_spec2 = {
    "pde": {}
}

start_time = time.time()
result2 = solve(case_spec2)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.3f} seconds")
print("Mesh resolution:", result2["solver_info"]["mesh_resolution"])
print("dt used:", result2["solver_info"]["dt"])
print("n_steps:", result2["solver_info"]["n_steps"])
print()

# Test 3: Partial time parameters
print("Test 3: Partial time parameters (only t_end)")
case_spec3 = {
    "pde": {
        "time": {
            "t_end": 0.05
        }
    }
}

start_time = time.time()
result3 = solve(case_spec3)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.3f} seconds")
print("Mesh resolution:", result3["solver_info"]["mesh_resolution"])
print("dt used:", result3["solver_info"]["dt"])
print("n_steps:", result3["solver_info"]["n_steps"])
