import numpy as np

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def f_source(x, t, kappa=1.0):
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    u_val = np.exp(-t) * np.exp(-40 * r2)
    # f = ∂u/∂t - κ∇²u
    # ∂u/∂t = -u
    # ∇²u = u*(6400*r² - 160)
    # f = -u - κ*u*(6400*r² - 160) = -u*(1 + κ*(6400*r² - 160))
    # With κ=1: f = -u*(6400*r² - 159)
    return -u_val * (1 + kappa * (6400 * r2 - 160))

# Test at a point
x = np.array([0.5, 0.5])
t = 0.1
print(f"At center (0.5, 0.5), t=0.1:")
print(f"u_exact = {u_exact(x, t)}")
print(f"f_source = {f_source(x, t)}")

# Test gradient
h = 1e-6
# ∂u/∂t ≈ (u(x, t+h) - u(x, t))/h
dudt = (u_exact(x, t+h) - u_exact(x, t))/h
print(f"∂u/∂t ≈ {dudt}")
print(f"-u = {-u_exact(x, t)}")

# Laplacian approximation
def laplacian_approx(x, t, h=1e-4):
    # ∇²u ≈ (u(x+h,y) + u(x-h,y) + u(x,y+h) + u(x,y-h) - 4u(x,y))/h²
    u_center = u_exact(x, t)
    u_xp = u_exact([x[0]+h, x[1]], t)
    u_xm = u_exact([x[0]-h, x[1]], t)
    u_yp = u_exact([x[0], x[1]+h], t)
    u_ym = u_exact([x[0], x[1]-h], t)
    return (u_xp + u_xm + u_yp + u_ym - 4*u_center)/(h*h)

lap = laplacian_approx(x, t)
print(f"∇²u ≈ {lap}")
print(f"u*(6400*r² - 160) = {u_exact(x, t) * (6400*0 - 160)}")

# Check if f = ∂u/∂t - ∇²u
f_computed = dudt - lap
print(f"\n∂u/∂t - ∇²u ≈ {f_computed}")
print(f"f_source = {f_source(x, t)}")
print(f"Difference: {abs(f_computed - f_source(x, t))}")
