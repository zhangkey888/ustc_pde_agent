import numpy as np
from solver import solve
import time

print("=== Final Comprehensive Test ===")

# Exact case from problem description
case_spec = {
    'pde': {
        'type': 'heat',
        'time': {
            't_end': 0.1,
            'dt': 0.01,
            'scheme': 'backward_euler'
        },
        'coefficients': {
            'kappa': 1.0
        },
        'manufactured_solution': 'exp(-t)*sin(pi*x)*sin(pi*y)'
    },
    'domain': {
        'type': 'unit_square',
        'bounds': [[0,1], [0,1]]
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"\nSolver took {end - start:.3f} seconds")
print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
print(f"Element degree: {result['solver_info']['element_degree']}")
print(f"Solver: {result['solver_info']['ksp_type']} with {result['solver_info']['pc_type']}")
print(f"Time stepping: dt={result['solver_info']['dt']}, n_steps={result['solver_info']['n_steps']}")
print(f"Total linear iterations: {result['solver_info']['iterations']}")

# Verify output shape
assert result['u'].shape == (50, 50), f"u shape is {result['u'].shape}, expected (50, 50)"
assert 'u_initial' in result, "Missing u_initial"
assert result['u_initial'].shape == (50, 50), f"u_initial shape is {result['u_initial'].shape}"

# Verify solver_info has all required fields
required_fields = ['mesh_resolution', 'element_degree', 'ksp_type', 'pc_type', 'rtol',
                   'iterations', 'dt', 'n_steps', 'time_scheme']
for field in required_fields:
    assert field in result['solver_info'], f"Missing {field} in solver_info"

# Compute error
u_grid = result['u']
nx, ny = u_grid.shape
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(u_grid - u_exact)
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(error)

print(f"\nAccuracy metrics:")
print(f"  L2 error: {l2_error:.3e} (required ≤ 2.98e-03)")
print(f"  Max error: {max_error:.3e}")
print(f"  Pass accuracy: {l2_error <= 2.98e-03}")

print(f"\nTime constraint: {end - start:.3f}s (required ≤ 13.651s)")
print(f"  Pass time: {end - start <= 13.651}")

print("\n=== All checks passed! ===")
