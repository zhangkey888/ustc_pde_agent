import numpy as np
import solver_degree2

# Modify solver to use degree 2
with open('solver_degree2.py', 'w') as f:
    with open('solver.py', 'r') as orig:
        content = orig.read()
        # Change element_degree from 1 to 2
        content = content.replace('element_degree = 1  # Linear elements (P1)', 'element_degree = 2  # Quadratic elements (P2)')
        f.write(content)

# Now test
import importlib
import sys
if 'solver_degree2' in sys.modules:
    importlib.reload(sys.modules['solver_degree2'])
else:
    import solver_degree2

case_spec = {
    "t_end": 0.1,
    "dt": 0.005,
    "scheme": "backward_euler"
}

result = solver_degree2.solve(case_spec)
u_grid = result["u"]

# Compute error
nx, ny = 50, 50
x_grid = np.linspace(0.0, 1.0, nx)
y_grid = np.linspace(0.0, 1.0, ny)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
u_exact = np.exp(-0.1) * np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)

dx = 1.0/(nx-1)
dy = 1.0/(ny-1)
error_grid = u_grid - u_exact
l2_error = np.sqrt(np.sum(error_grid**2) * dx * dy)
max_error = np.max(np.abs(error_grid))

print(f"Degree 2 results:")
print(f"L2 error on 50x50 grid: {l2_error:.6e}")
print(f"Max error on 50x50 grid: {max_error:.6e}")
print(f"Accuracy requirement: ≤ 1.22e-03")
print(f"L2 error meets requirement: {l2_error <= 1.22e-3}")
print(f"Max error meets requirement: {max_error <= 1.22e-3}")
print(f"Solver info: {result['solver_info']}")
