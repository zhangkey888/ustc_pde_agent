from mpi4py import MPI
from solver import solve
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

case_spec = {
    "pde": {
        "type": "parabolic",
        "time": {
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler"
        }
    }
}

result = solve(case_spec)

if rank == 0:
    # Quick verification
    u = result["u"]
    solver_info = result["solver_info"]
    
    # Check shape
    assert u.shape == (50, 50), f"Wrong shape: {u.shape}"
    
    # Check accuracy
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    max_error = np.max(np.abs(u - u_exact))
    
    print(f"Final test:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Required: ≤ 1.42e-03")
    print(f"  Pass: {max_error <= 1.42e-03}")
    
    # Check required fields
    assert "u" in result
    assert "solver_info" in result
    assert "u_initial" in result  # Optional but we provide it
    
    required_info = ["mesh_resolution", "element_degree", "ksp_type", "pc_type", "rtol", 
                     "iterations", "dt", "n_steps", "time_scheme"]
    for field in required_info:
        assert field in solver_info, f"Missing field: {field}"
    
    print(f"  All required fields present: ✓")
    print(f"  Solver info: mesh_resolution={solver_info['mesh_resolution']}, "
          f"element_degree={solver_info['element_degree']}, "
          f"dt={solver_info['dt']}, n_steps={solver_info['n_steps']}")
    
    if max_error <= 1.42e-03:
        print("\n✅ Solver meets all requirements!")
    else:
        print("\n❌ Solver fails accuracy requirement")

comm.Barrier()
