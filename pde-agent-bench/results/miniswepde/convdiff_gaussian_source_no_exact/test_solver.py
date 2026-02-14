import solver
import numpy as np

# Test case from problem description
case_spec = {
    "epsilon": 0.02,
    "beta": [8.0, 3.0],
    "pde": {
        "type": "convection-diffusion"
    }
}

result = solver.solve(case_spec)

# Check required fields
assert "u" in result
assert "solver_info" in result

u_grid = result["u"]
solver_info = result["solver_info"]

# Check u_grid shape
assert u_grid.shape == (50, 50), f"Expected shape (50, 50), got {u_grid.shape}"

# Check solver_info required fields
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", "iterations"]
for field in required_fields:
    assert field in solver_info, f"Missing field: {field}"

# Check values
assert solver_info["mesh_resolution"] in [32, 64, 128]
assert solver_info["element_degree"] == 1
assert solver_info["rtol"] == 1e-8
assert solver_info["iterations"] > 0

print("All tests passed!")
print(f"Mesh resolution: {solver_info['mesh_resolution']}")
print(f"Element degree: {solver_info['element_degree']}")
print(f"Solver type: {solver_info['ksp_type']}/{solver_info['pc_type']}")
print(f"Iterations: {solver_info['iterations']}")
print(f"Time: {solver_info.get('wall_time_sec', 'N/A'):.3f} seconds")
print(f"Solution min/max: {u_grid.min():.6f}, {u_grid.max():.6f}")
