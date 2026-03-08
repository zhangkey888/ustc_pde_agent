import time
import numpy as np
from solver import solve

print("=== Test 1: Default case (k=15) ===")
case1 = {"pde": {"k": 15.0}, "domain": {"bounds": [[0,0],[1,1]]}}
start = time.time()
result1 = solve(case1)
t1 = time.time() - start
print(f"Time: {t1:.3f}s")
print(f"Mesh: {result1['solver_info']['mesh_resolution']}")
print(f"Degree: {result1['solver_info']['element_degree']}")
u1 = result1['u']
print(f"Solution range: [{u1.min():.3e}, {u1.max():.3e}]")

print("\n=== Test 2: Smaller k (k=5) ===")
case2 = {"pde": {"k": 5.0}, "domain": {"bounds": [[0,0],[1,1]]}}
start = time.time()
result2 = solve(case2)
t2 = time.time() - start
print(f"Time: {t2:.3f}s")
print(f"Mesh: {result2['solver_info']['mesh_resolution']}")
print(f"Degree: {result2['solver_info']['element_degree']}")
u2 = result2['u']
print(f"Solution range: [{u2.min():.3e}, {u2.max():.3e}]")

print("\n=== Test 3: Larger k (k=30) ===")
case3 = {"pde": {"k": 30.0}, "domain": {"bounds": [[0,0],[1,1]]}}
start = time.time()
result3 = solve(case3)
t3 = time.time() - start
print(f"Time: {t3:.3f}s")
print(f"Mesh: {result3['solver_info']['mesh_resolution']}")
print(f"Degree: {result3['solver_info']['element_degree']}")
u3 = result3['u']
print(f"Solution range: [{u3.min():.3e}, {u3.max():.3e}]")

print("\n=== Summary ===")
print(f"All tests completed successfully.")
print(f"Times: {t1:.3f}, {t2:.3f}, {t3:.3f} seconds")
print(f"All within 14.614s limit: {t1<14.614 and t2<14.614 and t3<14.614}")

# Check that solutions are different (sanity)
diff12 = np.max(np.abs(u1 - u2))
diff13 = np.max(np.abs(u1 - u3))
print(f"\nMax difference between k=15 and k=5: {diff12:.3e}")
print(f"Max difference between k=15 and k=30: {diff13:.3e}")
print("(Should be non-zero, indicating solver responds to parameter changes)")

# Run the built-in test
print("\n=== Running solver.py __main__ block ===")
exec(open('solver.py').read())
