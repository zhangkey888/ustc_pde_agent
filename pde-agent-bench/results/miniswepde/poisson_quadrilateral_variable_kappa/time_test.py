import time
import solver
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.5*cos(2*pi*x)*cos(2*pi*y)"}
        }
    }
}
start = time.perf_counter()
result = solver.solve(case_spec)
end = time.perf_counter()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Time constraint: < 2.674s")
if end - start < 2.674:
    print("PASS time constraint")
else:
    print("FAIL time constraint")
