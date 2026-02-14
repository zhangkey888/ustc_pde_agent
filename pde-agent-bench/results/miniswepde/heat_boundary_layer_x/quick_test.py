import time
import solver

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.08,
            "dt": 0.008,
            "scheme": "backward_euler"
        }
    }
}

start = time.time()
result = solver.solve(case_spec)
elapsed = time.time() - start

print(f"Time: {elapsed:.2f}s")
print(f"Mesh: {result['solver_info']['mesh_resolution']}")
print(f"Degree: {result['solver_info']['element_degree']}")
print(f"Steps: {result['solver_info']['n_steps']}")
print(f"dt: {result['solver_info']['dt']}")
