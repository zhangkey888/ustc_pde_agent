import numpy as np
from mpi4py import MPI
import sys
sys.path.insert(0, '.')
from solver_fixed import solve

comm = MPI.COMM_WORLD
rank = comm.rank

case_spec = {'pde': {'time': {'t_end': 0.12, 'dt': 0.02}}}
result = solve(case_spec)

if rank == 0:
    print(f'Mesh resolution: {result["solver_info"]["mesh_resolution"]}')
    print(f'Solution shape: {result["u"].shape}')
    print(f'Solution min/max: {result["u"].min():.2e}, {result["u"].max():.2e}')
    print(f'Has NaN: {np.any(np.isnan(result["u"]))}')
    print(f'Initial shape: {result["u_initial"].shape}')
    print(f'Initial min/max: {result["u_initial"].min():.2e}, {result["u_initial"].max():.2e}')
    
    # Check that all values are finite
    assert np.all(np.isfinite(result["u"]))
    assert np.all(np.isfinite(result["u_initial"]))
    assert result["u"].shape == (50, 50)
    assert result["u_initial"].shape == (50, 50)
    print("All tests passed!")
