import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc

# Test the manufactured solution
comm = MPI.COMM_WORLD
rank = comm.rank

domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], 
                               [32, 32], cell_type=mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 2))

x = ufl.SpatialCoordinate(domain)
t = 0.4  # Final time
u_exact_expr = ufl.exp(-t) * (ufl.exp(x[0]) * ufl.sin(np.pi * x[1]))

# Create exact solution function
u_exact = fem.Function(V)
u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

# Compute L2 error if we had a numerical solution
# For now just check the exact solution evaluation
norm_form = fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)
norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))

if rank == 0:
    print(f"L2 norm of exact solution at t={t}: {norm_value}")
    
    # Check some point values
    # At (0,0.5): u = exp(-0.4)*(exp(0)*sin(pi*0.5)) = exp(-0.4)*1 = exp(-0.4) ≈ 0.67032
    # At (1,0.5): u = exp(-0.4)*(exp(1)*sin(pi*0.5)) = exp(-0.4)*exp(1) = exp(0.6) ≈ 1.82212
    print(f"Expected u(0,0.5) ≈ {np.exp(-0.4):.5f}")
    print(f"Expected u(1,0.5) ≈ {np.exp(0.6):.5f}")
