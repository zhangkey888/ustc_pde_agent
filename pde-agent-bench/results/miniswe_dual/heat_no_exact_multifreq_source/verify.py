import numpy as np

# Exact solution for heat equation with source
# du/dt - kappa * laplacian(u) = f
# f = sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)
# u(x,0) = 0, u = 0 on boundary
# 
# For mode A*sin(n*pi*x)*sin(m*pi*y):
#   eigenvalue lambda = kappa * (n^2 + m^2) * pi^2
#   u_mode(x,y,t) = (A/lambda) * (1 - exp(-lambda*t)) * sin(n*pi*x)*sin(m*pi*y)

kappa = 1.0
t_end = 0.12

# Mode 1: A=1, n=5, m=3
n1, m1, A1 = 5, 3, 1.0
lam1 = kappa * (n1**2 + m1**2) * np.pi**2
coeff1 = A1 / lam1 * (1 - np.exp(-lam1 * t_end))

# Mode 2: A=0.5, n=9, m=7
n2, m2, A2 = 9, 7, 0.5
lam2 = kappa * (n2**2 + m2**2) * np.pi**2
coeff2 = A2 / lam2 * (1 - np.exp(-lam2 * t_end))

print(f"Mode 1: lambda={lam1:.4f}, coeff={coeff1:.6e}")
print(f"Mode 2: lambda={lam2:.4f}, coeff={coeff2:.6e}")

# Evaluate on 50x50 grid
nx, ny = 50, 50
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
XX, YY = np.meshgrid(xs, ys, indexing='ij')

u_exact = (coeff1 * np.sin(n1*np.pi*XX) * np.sin(m1*np.pi*YY)
         + coeff2 * np.sin(n2*np.pi*XX) * np.sin(m2*np.pi*YY))

print(f"Exact u min: {u_exact.min():.6e}, max: {u_exact.max():.6e}")

# Now load solver result and compare
import sys
sys.path.insert(0, '.')
from solver import solve

case_spec = {
    "pde": {
        "type": "heat",
        "coefficients": {"kappa": 1.0},
        "source": "sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)",
        "initial_condition": "0.0",
        "time": {
            "t_end": 0.12,
            "dt": 0.02,
            "scheme": "backward_euler"
        },
        "boundary_conditions": {"type": "dirichlet", "value": 0.0}
    }
}

import time
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_num = result['u']
print(f"\nNumerical u min: {u_num.min():.6e}, max: {u_num.max():.6e}")

# Compute L2-like error on the grid
diff = u_num - u_exact
l2_err = np.sqrt(np.mean(diff**2))
linf_err = np.max(np.abs(diff))
rel_l2 = l2_err / (np.sqrt(np.mean(u_exact**2)) + 1e-15)

print(f"\nL2 error (RMSE): {l2_err:.6e}")
print(f"Linf error: {linf_err:.6e}")
print(f"Relative L2 error: {rel_l2:.6e}")
print(f"Wall time: {elapsed:.3f}s")
print(f"\nThreshold: 5.34e-02")
print(f"PASS: {l2_err < 5.34e-2}")
