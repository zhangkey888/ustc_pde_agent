import numpy as np
from solver import solve

result = solve({
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
})

my_u = result['u']
print(f"My solution min/max/mean: {my_u.min():.6f}, {my_u.max():.6f}, {my_u.mean():.6f}")
print(f"My solution at center (25,25): {my_u[25,25]:.6f}")

# Check a few points
nx, ny = 50, 50
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
for i in [0, 12, 25, 37, 49]:
    for j in [0, 12, 25, 37, 49]:
        print(f"({i},{j}) x={x[i]:.3f}, y={y[j]:.3f}: my={my_u[i,j]:.6f}")
