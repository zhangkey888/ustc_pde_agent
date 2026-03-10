import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

# Test basic functionality
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 4, 4)
V = fem.functionspace(domain, ("Lagrange", 1))
print("Test successful: created mesh and function space")
print(f"Mesh cells: {domain.topology.index_map(domain.topology.dim).size_local}")
print(f"DOFs: {V.dofmap.index_map.size_local}")
