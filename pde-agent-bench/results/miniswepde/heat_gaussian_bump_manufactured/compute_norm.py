import numpy as np

def u_exact(x, y, t):
    return np.exp(-t) * np.exp(-40 * ((x - 0.5)**2 + (y - 0.5)**2))

# Monte Carlo integration
np.random.seed(0)
N = 1000000
x = np.random.rand(N)
y = np.random.rand(N)
t = 0.1
vals = u_exact(x, y, t)
norm_sq = np.mean(vals**2)  # average over unit square
norm = np.sqrt(norm_sq)
print(f"Monte Carlo L2 norm: {norm:.6f}")
# Also compute integral of u^2
integral = np.mean(vals**2)
print(f"Integral of u^2: {integral:.6f}")
# True integral over R^2: ∫ exp(-80*r^2) dA = π/80 ≈ 0.03927
# Over unit square, slightly less.
print(f"π/80: {np.pi/80:.6f}")
