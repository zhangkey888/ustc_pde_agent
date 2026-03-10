import numpy as np
from mpi4py import MPI
import time
import solver

case_spec = {
    "pde": {
        "time": {
            "t_end": 0.05,
            "dt": 0.005,
            "scheme": "backward_euler"
        },
        "coefficients": {
            "kappa": 10.0
        }
    }
}

print(f"Rank {MPI.COMM_WORLD.rank}: Starting solver")
start = time.time()
result = solver.solve(case_spec)
end = time.time()

if MPI.COMM_WORLD.rank == 0:
    print(f"Time: {end - start:.3f} s")
    print(f"Mesh: {result['solver_info']['mesh_resolution']}")
    print(f"Error check: shape = {result['u'].shape}")
