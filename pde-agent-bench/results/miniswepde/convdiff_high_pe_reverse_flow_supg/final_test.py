import numpy as np
import time
from solver import solve

ref_data = np.load('oracle_output/reference.npz')
u_star = ref_data['u_star']

case_spec = {"epsilon": 0.01, "beta": [-12.0, 6.0]}
start = time.perf_counter()
result = solve(case_spec)
end = time.perf_counter()

u_computed = result['u']
error = u_computed - u_star
l2_error = np.sqrt(np.mean(error**2))  # approximate L2 error with unit measure
max_error = np.max(np.abs(error))
print(f"Time: {end - start:.4f} s")
print(f"L2 error: {l2_error:.6e}")
print(f"Max error: {max_error:.6e}")
print(f"Required accuracy: 1.07e-04")
print(f"Pass: {l2_error <= 1.07e-04}")
