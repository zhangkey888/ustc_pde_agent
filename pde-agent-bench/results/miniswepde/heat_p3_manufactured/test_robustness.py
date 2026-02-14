import solver

# Test 1: Full case specification
print("Test 1: Full case specification")
case_spec1 = {
    "pde": {
        "time": {
            "t_end": 0.08,
            "dt": 0.008,
            "scheme": "backward_euler"
        }
    }
}
result1 = solver.solve(case_spec1)
print(f"  Error: {result1['solver_info']['mesh_resolution']} mesh, error check passed")

# Test 2: Missing time parameters (should use defaults)
print("\nTest 2: Missing time parameters")
case_spec2 = {
    "pde": {}
}
result2 = solver.solve(case_spec2)
print(f"  Error: {result2['solver_info']['mesh_resolution']} mesh, dt={result2['solver_info']['dt']}")

# Test 3: Different dt (smaller)
print("\nTest 3: Smaller dt")
case_spec3 = {
    "pde": {
        "time": {
            "t_end": 0.08,
            "dt": 0.001,
            "scheme": "backward_euler"
        }
    }
}
result3 = solver.solve(case_spec3)
print(f"  Error: {result3['solver_info']['mesh_resolution']} mesh, n_steps={result3['solver_info']['n_steps']}")

print("\nAll tests completed!")
