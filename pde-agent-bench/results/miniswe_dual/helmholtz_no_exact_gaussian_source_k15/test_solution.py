import numpy as np
from solver import solve

case_spec = {
    "pde": {"k": 15.0, "type": "helmholtz"},
    "domain": {"bounds": [[0, 0], [1, 1]]}
}

result = solve(case_spec)
u = result["u"]

print("Solution analysis:")
print(f"Shape: {u.shape}")
print(f"Min: {u.min():.6e}, Max: {u.max():.6e}")
print(f"Mean: {u.mean():.6e}, Std: {u.std():.6e}")

# Check symmetry: source at (0.35, 0.55), domain is symmetric
# Solution should have some structure
print("\nValues along diagonal from (0,0) to (1,1):")
for i in range(0, 50, 10):
    val = u[i, i]
    print(f"  u[{i},{i}] = {val:.6e}")

# Check near source location (0.35, 0.55)
# Convert to indices: 0.35*49 ≈ 17, 0.55*49 ≈ 27
i, j = int(0.35*49), int(0.55*49)
print(f"\nNear source point (0.35, 0.55) -> indices ({i},{j}):")
print(f"  u[{i},{j}] = {u[i,j]:.6e}")
print(f"  u[{i-1},{j}] = {u[i-1,j]:.6e}")
print(f"  u[{i+1},{j}] = {u[i+1,j]:.6e}")

# Quick residual check: approximate Laplacian using finite differences
# This is rough but can indicate if solution is reasonable
h = 1.0 / 49
laplacian = np.zeros_like(u)
laplacian[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]) / (h*h)

# Helmholtz residual: -laplacian - k^2 u - f
# We don't have f on the grid, but we can check order of magnitude
k = 15.0
helmholtz_res = -laplacian - k*k*u
print(f"\nHelmholtz residual (approx FD Laplacian):")
print(f"  Max |res|: {np.max(np.abs(helmholtz_res)):.6e}")
print(f"  Mean |res|: {np.mean(np.abs(helmholtz_res)):.6e}")

# The residual should be small where solution is smooth
print("\nResidual should be small (order ~1e-3 to 1e-1 is okay for this grid)")
