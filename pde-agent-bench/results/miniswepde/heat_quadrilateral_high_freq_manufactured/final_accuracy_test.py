import numpy as np
from solver import solve

result = solve({'pde': {'time': {'t_end': 0.1, 'dt': 0.005}}})

# Compute exact solution on the same 50x50 grid
nx, ny = 50, 50
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

def exact(x, y, t):
    return np.exp(-t) * np.sin(4*np.pi*x) * np.sin(4*np.pi*y)

u_exact = exact(X, Y, 0.1)
u_computed = result['u']

error = np.abs(u_computed - u_exact)
max_error = np.max(error)
l2_error = np.sqrt(np.mean(error**2))

print(f"Max error vs exact solution: {max_error:.2e}")
print(f"L2 error vs exact solution: {l2_error:.2e}")
print(f"Accuracy requirement: 9.19e-03")
print(f"Pass: {max_error <= 9.19e-03}")

# Also check time
print(f"\nSolve time from solver_info not directly available, but should be < 26.313s")
print(f"Based on previous runs: ~1.2s for dt=0.005, ~24s for dt=0.001")
