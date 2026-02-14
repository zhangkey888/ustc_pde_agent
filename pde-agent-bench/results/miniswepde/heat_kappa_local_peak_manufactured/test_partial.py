from solver import solve

# Test with partial time parameters
case_spec = {
    "pde": {
        "time": {
            "dt": 0.005,  # Only dt provided, t_end should use default
            "scheme": "backward_euler"
        }
    }
}
result = solve(case_spec)
print(f"Solution shape: {result['u'].shape}")
print(f"Solver info: {result['solver_info']}")
print(f"dt used: {result['solver_info']['dt']}")
print(f"n_steps: {result['solver_info']['n_steps']}")
print(f"t_end implied: {result['solver_info']['dt'] * result['solver_info']['n_steps']}")
