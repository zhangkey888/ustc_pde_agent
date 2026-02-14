import solver
import numpy as np
import time

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.12,
            "dt": 0.03,
            "scheme": "backward_euler"
        }
    }
}

print("Running final test with problem parameters...")
start = time.time()
result = solver.solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Time steps: {result['solver_info']['n_steps']}")
print(f"Total linear iterations: {result['solver_info']['iterations']}")
print(f"Solver type: {result['solver_info']['ksp_type']} with {result['solver_info']['pc_type']}")
print(f"Solution shape: {result['u'].shape}")
print(f"Solution min/max: {np.min(result['u']):.6f}, {np.max(result['u']):.6f}")
print(f"Initial condition min/max: {np.min(result['u_initial']):.6f}, {np.max(result['u_initial']):.6f}")

# Check for NaN values
if np.any(np.isnan(result['u'])):
    print("WARNING: Solution contains NaN values!")
else:
    print("Solution contains no NaN values.")

# Check time constraint
if result['solver_info']['wall_time_sec'] <= 33.684:
    print("PASS: Time constraint met.")
else:
    print(f"FAIL: Time constraint not met. {result['solver_info']['wall_time_sec']:.2f} > 33.684")

print("\nSolver info keys:", list(result['solver_info'].keys()))
