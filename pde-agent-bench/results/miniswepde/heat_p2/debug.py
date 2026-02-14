import numpy as np
from solver import solve

case_spec = {
    'pde': {
        'time': {
            't_end': 0.06,
            'dt': 0.01,
            'scheme': 'backward_euler'
        }
    },
    'coefficients': {
        'kappa': 1.0
    },
    'domain': {
        'bounds': [[0.0, 0.0], [1.0, 1.0]]
    }
}

result = solve(case_spec)
u_grid = result['u']

# Exact solution at t=0.06
def exact(x, y):
    return np.exp(-0.06) * (x**2 + y**2)

# Sample at a few points
points = [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.2, 0.8)]
nx, ny = u_grid.shape
x_vals = np.linspace(0.0, 1.0, nx)
y_vals = np.linspace(0.0, 1.0, ny)

for (px, py) in points:
    # Find closest grid indices
    i = np.argmin(np.abs(x_vals - px))
    j = np.argmin(np.abs(y_vals - py))
    u_num = u_grid[i, j]
    u_ex = exact(px, py)
    print(f"Point ({px:.2f}, {py:.2f}): numerical={u_num:.6f}, exact={u_ex:.6f}, error={abs(u_num-u_ex):.2e}")
