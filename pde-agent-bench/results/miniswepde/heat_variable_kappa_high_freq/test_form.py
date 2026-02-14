import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# Create a simple mesh
domain = mesh.create_unit_square(comm, 8, 8, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Simple test form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

# Simple Poisson: ∇u·∇v dx
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_form = fem.form(a)

print(f"Rank {rank}: Form created successfully")
print(f"Rank {rank}: Form function spaces: {a_form.function_spaces}")

# Try to assemble matrix
try:
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    print(f"Rank {rank}: Matrix assembly successful, size: {A.getSize()}")
except Exception as e:
    print(f"Rank {rank}: Matrix assembly failed: {e}")
    
# Now test with a form that includes a coefficient
kappa = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
a2 = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
a2_form = fem.form(a2)

print(f"Rank {rank}: Form with coefficient created")

try:
    A2 = petsc.assemble_matrix(a2_form)
    A2.assemble()
    print(f"Rank {rank}: Matrix with coefficient assembly successful, size: {A2.getSize()}")
except Exception as e:
    print(f"Rank {rank}: Matrix with coefficient assembly failed: {e}")
