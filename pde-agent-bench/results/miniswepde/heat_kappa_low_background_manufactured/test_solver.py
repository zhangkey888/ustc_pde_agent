import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from solver import solve
import time

# Create case_spec
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

# Time the solve
start = time.time()
result = solve(case_spec)
end = time.time()
print(f"Solve time: {end - start:.3f} s")

# Compute error against exact solution
u_grid = result["u"]
nx, ny = u_grid.shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
t_end = 0.1
u_exact = np.exp(-t_end) * np.sin(np.pi * X) * np.sin(np.pi * Y)

error = np.abs(u_grid - u_exact)
l2_error = np.sqrt(np.mean(error**2))
max_error = np.max(error)
print(f"L2 error: {l2_error:.2e}")
print(f"Max error: {max_error:.2e}")
print(f"Accuracy requirement: ≤ 4.07e-03")
print(f"Time requirement: ≤ 11.051 s")
