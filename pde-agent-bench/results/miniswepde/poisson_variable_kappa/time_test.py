import time
from solver import solve
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "1 + 0.5*sin(2*pi*x)*sin(2*pi*y)"}
        }
    }
}
start = time.perf_counter()
result = solve(case_spec)
end = time.perf_counter()
print(f"Time taken: {end - start:.3f} seconds")
print(f"Time limit: 2.742 seconds")
print(f"Within limit? {end - start <= 2.742}")
