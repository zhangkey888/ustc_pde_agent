import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem

comm = MPI.COMM_WORLD

def u_exact(x, t):
    return np.exp(-t) * np.exp(-40 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

for N in [32, 64, 128]:
    for degree in [1, 2]:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        t = 0.1
        u_fe = fem.Function(V)
        u_fe.interpolate(lambda x: u_exact(x, t))
        
        # Get dof coordinates
        dof_coords = V.tabulate_dof_coordinates()
        
        # Compute exact values at dof coordinates
        exact_vals = np.zeros_like(dof_coords[:, 0])
        for i, coord in enumerate(dof_coords):
            x_arr = np.array([[coord[0]], [coord[1]], [0.0]])
            exact_vals[i] = u_exact(x_arr, t)
        
        # Compare
        error = u_fe.x.array - exact_vals
        l2_error = np.sqrt(np.mean(error**2))
        print(f"N={N}, degree={degree}: Interpolation L2 error = {l2_error:.6e}")
        
        # Also check max
        max_error = np.max(np.abs(error))
        print(f"  Max interpolation error = {max_error:.6e}")
