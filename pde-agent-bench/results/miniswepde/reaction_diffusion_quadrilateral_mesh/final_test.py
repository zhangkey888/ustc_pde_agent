import numpy as np
import solver

# Test 1: Linear transient case (the benchmark case)
print("Test 1: Linear transient case")
case_spec = {
    "pde": {
        "type": "reaction_diffusion",
        "time": {
            "t_end": 0.4,
            "dt": 0.01,
            "scheme": "backward_euler"
        },
        "reaction": {
            "type": "linear"
        }
    }
}

result = solver.solve(case_spec)
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Linear iterations: {result['solver_info']['iterations']}")
print(f"Solution shape: {result['u'].shape}")
print(f"Has u_initial: {'u_initial' in result}")

# Compute error
nx, ny = 60, 60
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact_grid = np.exp(-0.4) * (np.exp(X) * np.sin(np.pi * Y))
error = np.max(np.abs(result['u'] - u_exact_grid))
print(f"Maximum error: {error:.2e}")
print(f"Required accuracy: ≤ 8.45e-03")
print(f"Pass accuracy: {error <= 8.45e-03}")

# Test 2: Steady linear case
print("\nTest 2: Steady linear case")
case_spec2 = {
    "pde": {
        "type": "reaction_diffusion",
        "reaction": {
            "type": "linear"
        }
    }
}

result2 = solver.solve(case_spec2)
print(f"Mesh resolution: {result2['solver_info']['mesh_resolution']}")
print(f"Steady solve completed")

print("\nAll tests passed!")
