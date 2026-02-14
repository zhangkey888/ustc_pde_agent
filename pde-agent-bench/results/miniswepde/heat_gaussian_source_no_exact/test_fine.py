import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import solver

comm = MPI.COMM_WORLD
rank = comm.rank

# Force N=128
N = 128
degree = 1
dt = 0.02
t_end = 0.1

if rank == 0:
    print(f"Testing fine mesh N={N}")

domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

# Call the internal function (need to expose it or copy code)
# Instead, we'll use the solve function but hack resolutions
import solver
# Temporarily modify solver.resolutions
original_solve = solver.solve

def forced_solve(case_spec):
    # Override the adaptive loop
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    t_end = 0.1
    dt_suggested = 0.02
    scheme = "backward_euler"
    
    if case_spec is not None:
        pde_info = case_spec.get('pde', {})
        time_info = pde_info.get('time', {})
        t_end = time_info.get('t_end', t_end)
        dt_suggested = time_info.get('dt', dt_suggested)
        scheme = time_info.get('scheme', scheme)
    
    degree = 1
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # We need to call solve_heat_on_mesh but it's not exposed
    # Instead, just call original solve but it will do adaptive loop
    # Let's just print that we're testing
    if rank == 0:
        print("Testing with forced N=128")
    
    # Use a monkey-patch: temporarily replace resolutions list
    original_resolutions = solver.resolutions if hasattr(solver, 'resolutions') else None
    solver.resolutions = [N]  # Only one resolution
    
    try:
        result = original_solve(case_spec)
    finally:
        if original_resolutions is not None:
            solver.resolutions = original_resolutions
    
    return result

# Monkey patch
solver.resolutions = [N]
case_spec = {}
result = solver.solve(case_spec)

if rank == 0:
    print(f"Success! mesh_resolution: {result['solver_info']['mesh_resolution']}")
    print(f"u shape: {result['u'].shape}")
    print(f"iterations: {result['solver_info']['iterations']}")
