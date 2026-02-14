import numpy as np
from mpi4py import MPI
from solver import solve

comm = MPI.COMM_WORLD
rank = comm.rank

# Test with different kappa
case_spec = {
    "pde": {
        "type": "poisson",
        "coefficients": {"kappa": 2.5}  # Different kappa
    }
}

result = solve(case_spec)

if rank == 0:
    print("\nTest with kappa=2.5")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution min/max: {result['u'].min():.6f}, {result['u'].max():.6f}")
