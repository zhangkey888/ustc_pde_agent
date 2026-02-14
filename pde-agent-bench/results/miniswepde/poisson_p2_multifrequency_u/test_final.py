import solver
import numpy as np

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 1.0}
    }
}
result = solver.solve(case_spec)

# Check output structure
assert "u" in result
assert "solver_info" in result
assert result["u"].shape == (50, 50)
assert isinstance(result["u"], np.ndarray)

required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for key in required_keys:
    assert key in result["solver_info"], f"Missing {key} in solver_info"

print("All checks passed!")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Solver: {result['solver_info']['ksp_type']} with {result['solver_info']['pc_type']}")
print(f"Iterations: {result['solver_info']['iterations']}")
print(f"u shape: {result['u'].shape}")
print(f"u dtype: {result['u'].dtype}")
