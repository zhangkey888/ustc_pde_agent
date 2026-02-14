import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Simple test: create mesh and function space
domain = mesh.create_rectangle(comm, [[0,0], [1,1]], [8, 8], cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Test SUPG parameter computation
x = ufl.SpatialCoordinate(domain)
beta = fem.Constant(domain, np.array([15.0, 0.0], dtype=np.float64))

# Instead of ufl.CellDiameter, use a characteristic length based on mesh
h = ufl.CellVolume(domain)**(1.0/domain.topology.dim)  # Approximate cell size
beta_norm = ufl.sqrt(ufl.dot(beta, beta))
tau = h / (2.0 * beta_norm + 1e-12)

print(f"Rank {rank}: tau type = {type(tau)}, h type = {type(h)}")

# Test if we can evaluate tau at a point
try:
    # Create a form to evaluate tau
    tau_form = fem.form(tau * ufl.dx)
    tau_val = fem.assemble_scalar(tau_form)
    print(f"Rank {rank}: Integrated tau = {tau_val}")
except Exception as e:
    print(f"Rank {rank}: Error evaluating tau: {e}")
