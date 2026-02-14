import numpy as np
from solver import solve

# Test case specification
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}

print("Testing adaptive mesh refinement...")
print("=" * 50)

# Run solver multiple times to see behavior
for test in range(3):
    print(f"\nTest run {test+1}:")
    result = solve(case_spec)
    solver_info = result["solver_info"]
    
    print(f"  Mesh resolution: {solver_info['mesh_resolution']}")
    print(f"  Solver type: {solver_info['ksp_type']}/{solver_info['pc_type']}")
    print(f"  Iterations: {solver_info['iterations']}")
    print(f"  Time: {solver_info['wall_time_sec']:.2f}s")
    
    # Compute error
    u_grid = result["u"]
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(4*np.pi*X) * np.sin(4*np.pi*Y)
    l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"  L2 error: {l2_error:.2e}")

print("\n" + "=" * 50)
print("Testing with different dt values...")

# Test with smaller dt
case_spec_small_dt = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.001,  # Smaller dt
            "scheme": "backward_euler"
        }
    }
}

result = solve(case_spec_small_dt)
solver_info = result["solver_info"]
print(f"\ndt=0.001:")
print(f"  Mesh resolution: {solver_info['mesh_resolution']}")
print(f"  n_steps: {solver_info['n_steps']}")
print(f"  Time: {solver_info['wall_time_sec']:.2f}s")

# Compute error
u_grid = result["u"]
l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
print(f"  L2 error: {l2_error:.2e}")
