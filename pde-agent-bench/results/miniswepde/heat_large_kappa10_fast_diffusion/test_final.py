import solver
import numpy as np

# Case spec as might be provided by evaluator
case_spec = {
    "pde": {
        "type": "heat",
        "time": {
            "t_end": 0.05,
            "dt": 0.005,
            "scheme": "backward_euler"
        },
        "coefficients": {
            "kappa": 10.0
        }
    }
}

result = solver.solve(case_spec)
print("Keys in result:", list(result.keys()))
print("u shape:", result["u"].shape)
print("u_initial shape:", result["u_initial"].shape)
print("Solver info:")
for k, v in result["solver_info"].items():
    print(f"  {k}: {v}")

# Check that u is not all zeros or NaN
print("\nChecking solution quality:")
print(f"u min: {np.min(result['u']):.6f}, max: {np.max(result['u']):.6f}, mean: {np.mean(result['u']):.6f}")
print(f"u_initial min: {np.min(result['u_initial']):.6f}, max: {np.max(result['u_initial']):.6f}")

# Quick error estimate (approximate)
# At t=0.05, u_exact ≈ 0.951 * sin(pi*x)*sin(pi*y)
# At center (0.5,0.5), exact ≈ 0.951
center_idx = 25, 25  # 50x50 grid, index 25 is 0.5
u_center = result["u"][center_idx]
print(f"\nu at center (0.5,0.5): {u_center:.6f}, expected ~0.951")
