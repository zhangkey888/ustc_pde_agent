import numpy as np
from solver import solve
case_spec = {'epsilon': 0.01, 'beta': [0.0, 15.0]}
result = solve(case_spec)
u_grid = result['u']
nx, ny = 50, 50
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
error = np.abs(u_grid - u_exact)
max_error = np.max(error)
mean_error = np.mean(error)
l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
print(f"Max error: {max_error:.2e}")
print(f"Mean error: {mean_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
print(f"Required accuracy: 4.18e-04")
print(f"Pass? {l2_error <= 4.18e-04}")
