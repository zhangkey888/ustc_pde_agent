import solver
import numpy as np
import time

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0},
        "source": "gaussian"
    }
}
start = time.time()
result = solver.solve(case_spec)
end = time.time()

print("=== Solver Results ===")
print(f"Time: {end-start:.3f} s")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"KSP type: {result['solver_info']['ksp_type']}")
print(f"PC type: {result['solver_info']['pc_type']}")
print(f"RTOL: {result['solver_info']['rtol']}")
print(f"Iterations: {result['solver_info']['iterations']}")
print(f"Solution shape: {result['u'].shape}")
print(f"Solution dtype: {result['u'].dtype}")
print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")
print(f"Solution mean: {result['u'].mean():.6f}")
print("\nChecking for NaN/inf...")
print(f"NaN count: {np.isnan(result['u']).sum()}")
print(f"Inf count: {np.isinf(result['u']).sum()}")
