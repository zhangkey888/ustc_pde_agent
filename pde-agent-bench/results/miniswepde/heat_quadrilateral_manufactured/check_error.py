import numpy as np
from solver import solve
import time

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
        }
    },
    'domain': {
        'type': 'unit_square',
        'bounds': [[0,1], [0,1]]
    }
}

start = time.time()
result = solve(case_spec)
end = time.time()

print(f"Time taken: {end - start:.3f} seconds")
print(f"Mesh resolution: {result['solver_info']['mesh_resolution']}")

# Compute error against exact solution
u_grid = result['u']
nx, ny = u_grid.shape
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Exact solution at t=0.1
u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u_grid - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error: {max_error:.3e}")
print(f"L2 error: {l2_error:.3e}")
print(f"Accuracy requirement: ≤ 2.98e-03")
print(f"Time requirement: ≤ 13.651s")
print(f"Pass accuracy: {l2_error <= 2.98e-03}")
print(f"Pass time: {end - start <= 13.651}")
