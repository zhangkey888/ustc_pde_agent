import time
import solver
case_spec = {'kappa': {'type': 'expr', 'expr': '1 + 15*exp(-200*((x-0.25)**2 + (y-0.25)**2)) + 15*exp(-200*((x-0.75)**2 + (y-0.75)**2))'}}
start = time.perf_counter()
result = solver.solve(case_spec)
end = time.perf_counter()
print(f"Solve time: {end-start:.6f} seconds")
print(f"Error check: L2 error should be < 4.88e-04")
print(f"Solver info: {result['solver_info']}")
