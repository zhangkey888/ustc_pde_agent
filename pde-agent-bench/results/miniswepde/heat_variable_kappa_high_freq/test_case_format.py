import solver

# Test with the nested format that might be used by evaluator
case_spec = {
    "pde": {
        "time": {
            "t_end": 0.1,
            "dt": 0.005,
            "scheme": "backward_euler"
        }
    }
}

result = solver.solve(case_spec)
print("Test with nested format passed")
print("Solver info:", result["solver_info"])

# Also test with direct format
case_spec2 = {
    "t_end": 0.1,
    "dt": 0.005,
    "scheme": "backward_euler"
}

result2 = solver.solve(case_spec2)
print("\nTest with direct format passed")
print("Solver info:", result2["solver_info"])
