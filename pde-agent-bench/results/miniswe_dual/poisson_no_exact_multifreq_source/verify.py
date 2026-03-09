import numpy as np

# Exact solution for -Laplacian(u) = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
# u_exact = sin(5*pi*x)*sin(3*pi*y) / (pi^2*(25+9)) + 0.5*sin(9*pi*x)*sin(7*pi*y) / (pi^2*(81+49))

nx, ny = 50, 50
xs = np.linspace(0.0, 1.0, nx)
ys = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(xs, ys, indexing='ij')

pi = np.pi
u_exact = (np.sin(5*pi*X)*np.sin(3*pi*Y) / (pi**2 * (25+9)) + 
           0.5*np.sin(9*pi*X)*np.sin(7*pi*Y) / (pi**2 * (81+49)))

print(f"Exact solution range: [{u_exact.min():.6e}, {u_exact.max():.6e}]")

# Now run solver and compare
from solver import solve
import time
t0 = time.time()
result = solve({})
elapsed = time.time() - t0

u_num = result["u"]
error = np.sqrt(np.mean((u_num - u_exact)**2))
max_error = np.max(np.abs(u_num - u_exact))
rel_error = error / np.sqrt(np.mean(u_exact**2))

print(f"Numerical range: [{u_num.min():.6e}, {u_num.max():.6e}]")
print(f"L2 error (RMSE): {error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Relative L2 error: {rel_error:.6e}")
print(f"Wall time: {elapsed:.3f}s")
print(f"Error threshold: 2.68e-02")
print(f"PASS: {error < 2.68e-02}")
