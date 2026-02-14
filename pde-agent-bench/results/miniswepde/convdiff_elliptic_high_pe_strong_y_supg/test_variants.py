import time
import numpy as np
from solver import solve

def run_test(epsilon, beta, label):
    case_spec = {'epsilon': epsilon, 'beta': beta}
    start = time.time()
    result = solve(case_spec)
    end = time.time()
    u_grid = result['u']
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    print(f"{label}: eps={epsilon}, beta={beta}")
    print(f"  time={end-start:.3f}s, mesh={result['solver_info']['mesh_resolution']}, iter={result['solver_info']['iterations']}, L2 err={l2_error:.2e}")
    return l2_error <= 4.18e-04 and end-start <= 2.504

# Test cases
run_test(0.01, [0.0, 15.0], "Original")
run_test(0.001, [0.0, 15.0], "Higher Pe")
run_test(0.1, [0.0, 15.0], "Lower Pe")
run_test(0.01, [15.0, 0.0], "Beta x-direction")
run_test(0.01, [10.0, 10.0], "Beta diagonal")
