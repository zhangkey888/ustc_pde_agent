import numpy as np
from mpi4py import MPI
from solver import solve

comm = MPI.COMM_WORLD
rank = comm.rank

# Test 1: Default parameters
if rank == 0:
    print("Test 1: Default parameters")
case_spec = {}
result1 = solve(case_spec)
if rank == 0:
    print(f"  Mesh resolution: {result1['solver_info']['mesh_resolution']}")
    print(f"  Solution shape: {result1['u'].shape}")
    assert result1['u'].shape == (50, 50)
    print("  ✓ Test 1 passed")

# Test 2: Provided parameters
if rank == 0:
    print("\nTest 2: Provided parameters")
case_spec2 = {
    "pde": {
        "time": {
            "t_end": 0.06,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}
result2 = solve(case_spec2)
if rank == 0:
    print(f"  Mesh resolution: {result2['solver_info']['mesh_resolution']}")
    print(f"  dt: {result2['solver_info']['dt']}")
    print(f"  n_steps: {result2['solver_info']['n_steps']}")
    assert abs(result2['solver_info']['dt'] * result2['solver_info']['n_steps'] - 0.06) < 1e-10
    print("  ✓ Test 2 passed")

# Test 3: Check solver_info fields
if rank == 0:
    print("\nTest 3: Check solver_info fields")
required_fields = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                   "iterations", "dt", "n_steps", "time_scheme"]
for field in required_fields:
    assert field in result1['solver_info'], f"Missing field: {field}"
if rank == 0:
    print("  ✓ All required fields present")

# Test 4: Check initial condition
if rank == 0:
    print("\nTest 4: Check initial condition")
    assert np.allclose(result1['u_initial'], 0.0, atol=1e-12)
    print("  ✓ Initial condition is zero")

if rank == 0:
    print("\nAll tests passed!")
