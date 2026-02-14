import solver
import numpy as np

# Test with transient case specification
test_spec = {
    "pde": {
        "type": "parabolic",
        "time": {"t_end": 0.1, "dt": 0.01}
    },
    "domain": {"shape": "square"},
    "coefficients": {
        "kappa": {"type": "expr", "expr": "1 + 0.3*sin(2*pi*x)*cos(2*pi*y)"}
    },
    "manufactured_solution": "sin(pi*x)*sin(pi*y)"
}

result = solver.solve(test_spec)
print("Transient test - Solution shape:", result["u"].shape)
print("Transient test - Solver info:", result["solver_info"])
print("Has u_initial:", "u_initial" in result)
if "u_initial" in result:
    print("u_initial shape:", result["u_initial"].shape)
