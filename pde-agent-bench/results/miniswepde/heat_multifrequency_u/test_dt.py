import solver
import time

# Test with smaller dt
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,  # smaller dt
            "scheme": "backward_euler"
        }
    }
}

print("Testing with dt=0.005")
start = time.time()
result1 = solver.solve(case_spec1)
end = time.time()
print(f"Time: {end-start:.2f}s")
print(f"Mesh: {result1['solver_info']['mesh_resolution']}")
print(f"Steps: {result1['solver_info']['n_steps']}")

# Test with larger dt
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.02,  # larger dt
            "scheme": "backward_euler"
        }
    }
}

print("\nTesting with dt=0.02")
start = time.time()
result2 = solver.solve(case_spec2)
end = time.time()
print(f"Time: {end-start:.2f}s")
print(f"Mesh: {result2['solver_info']['mesh_resolution']}")
print(f"Steps: {result2['solver_info']['n_steps']}")

# Test with minimal case_spec (should use defaults)
case_spec3 = {}
print("\nTesting with empty case_spec")
start = time.time()
result3 = solver.solve(case_spec3)
end = time.time()
print(f"Time: {end-start:.2f}s")
print(f"Mesh: {result3['solver_info']['mesh_resolution']}")
print(f"Steps: {result3['solver_info']['n_steps']}")
print(f"dt: {result3['solver_info']['dt']}")
