import time
import solver

case_spec = {
    "pde": {
        "type": "elliptic"
    }
}

start_time = time.time()
result = solver.solve(case_spec)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.3f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Total iterations: {result['solver_info']['iterations']}")
