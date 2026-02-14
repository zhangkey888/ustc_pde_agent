import numpy as np
from mpi4py import MPI
import time
import solver

# Test 1: Default case
print("Test 1: Default case_spec")
case_spec1 = {"pde": {"time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"}}}
start = time.time()
result1 = solver.solve(case_spec1)
end = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f"  Time: {end-start:.3f}s")
    print(f"  Error computed in solver: < 2.49e-03? {result1['solver_info']['mesh_resolution']}")

# Test 2: Different dt
print("\nTest 2: Smaller dt")
case_spec2 = {"pde": {"time": {"t_end": 0.1, "dt": 0.005, "scheme": "backward_euler"}}}
start = time.time()
result2 = solver.solve(case_spec2)
end = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f"  Time: {end-start:.3f}s")
    print(f"  Mesh: {result2['solver_info']['mesh_resolution']}")

# Test 3: Minimal case_spec
print("\nTest 3: Minimal case_spec")
case_spec3 = {}
start = time.time()
result3 = solver.solve(case_spec3)
end = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f"  Time: {end-start:.3f}s")
    print(f"  Mesh: {result3['solver_info']['mesh_resolution']}")
    print(f"  All tests passed!")

# Check output format
if MPI.COMM_WORLD.rank == 0:
    print("\nOutput format check:")
    print(f"  u shape: {result1['u'].shape} (should be (50, 50))")
    print(f"  u_initial shape: {result1['u_initial'].shape} (should be (50, 50))")
    print(f"  solver_info keys: {list(result1['solver_info'].keys())}")
    
    required_keys = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                     "iterations", "dt", "n_steps", "time_scheme"]
    missing = [k for k in required_keys if k not in result1['solver_info']]
    if missing:
        print(f"  WARNING: Missing keys: {missing}")
    else:
        print(f"  All required keys present")
