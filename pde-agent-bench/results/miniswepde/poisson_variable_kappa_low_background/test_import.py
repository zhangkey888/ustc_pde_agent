import sys
sys.path.insert(0, '.')
from solver import solve

case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {
            "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
        }
    }
}

result = solve(case_spec)
assert "u" in result
assert "solver_info" in result
assert result["u"].shape == (50, 50)
print("Import test passed.")
