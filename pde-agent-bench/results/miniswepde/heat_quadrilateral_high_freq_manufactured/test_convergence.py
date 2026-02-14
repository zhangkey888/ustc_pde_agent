import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

# Test the convergence check logic
comm = MPI.COMM_WORLD
resolutions = [32, 64, 128]
prev_norm = None

for N in resolutions:
    domain = mesh.create_unit_square(comm, N, N)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Simple test function
    u = fem.Function(V)
    u.interpolate(lambda x: np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]))
    
    norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
    norm_value = np.sqrt(fem.assemble_scalar(norm_form))
    
    print(f"N={N}, norm={norm_value:.6f}")
    
    if prev_norm is not None:
        relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
        print(f"  Relative error vs previous: {relative_error:.6f}")
        if relative_error < 0.01:
            print(f"  CONVERGED at N={N}")
            break
    
    prev_norm = norm_value
