from solver import solve

# Test with minimal case_spec
case_spec = {}
result = solve(case_spec)
print(f"Solution shape: {result['u'].shape}")
print(f"Solver info: {result['solver_info']}")
print(f"dt used: {result['solver_info']['dt']}")
print(f"t_end implied: {result['solver_info']['dt'] * result['solver_info']['n_steps']}")
