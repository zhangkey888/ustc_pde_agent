import numpy as np

def u_exact(x, y, t):
    return np.exp(-t) * np.exp(-40 * ((x - 0.5)**2 + (y - 0.5)**2))

def laplacian_u(x, y, t):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    u = u_exact(x, y, t)
    return u * (6400 * r2 - 160)

def u_t(x, y, t):
    u = u_exact(x, y, t)
    return -u

def f_exact(x, y, t):
    kappa = 1.0
    return u_t(x, y, t) - kappa * laplacian_u(x, y, t)

# Test at a few points
test_points = [(0.5, 0.5), (0.6, 0.5), (0.5, 0.6), (0.3, 0.7)]
t = 0.1

print("Testing manufactured solution at t=0.1:")
print("(x, y)       u           ∂u/∂t       ∇²u         f=∂u/∂t-∇²u")
print("-" * 60)
for x, y in test_points:
    u = u_exact(x, y, t)
    ut = u_t(x, y, t)
    lap = laplacian_u(x, y, t)
    f = f_exact(x, y, t)
    print(f"({x:.1f},{y:.1f})   {u:.6f}   {ut:.6f}   {lap:.6f}   {f:.6f}")

# Check PDE: ∂u/∂t - ∇²u should equal f
print("\nChecking PDE satisfaction:")
for x, y in test_points:
    lhs = u_t(x, y, t) - laplacian_u(x, y, t)
    rhs = f_exact(x, y, t)
    print(f"({x:.1f},{y:.1f}): ∂u/∂t-∇²u={lhs:.6f}, f={rhs:.6f}, diff={abs(lhs-rhs):.2e}")
