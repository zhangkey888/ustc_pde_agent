import numpy as np
import time

# Test different parameter combinations
params = [
    (64, 3, 2),
    (72, 3, 2),
    (80, 3, 2),
    (48, 4, 3),
    (56, 4, 3),
]

for N, du, dp in params:
    # Quick estimate: for P3/P2, DOFs ~ 2*(N*du+1)^2 + (N*dp+1)^2
    vel_dofs = 2 * (N*du + 1)**2  # rough
    pres_dofs = (N*dp + 1)**2
    total = vel_dofs + pres_dofs
    print(f"N={N}, P{du}/P{dp}: ~{total} DOFs")
