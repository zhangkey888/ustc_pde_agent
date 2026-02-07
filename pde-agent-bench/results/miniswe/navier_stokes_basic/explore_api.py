import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
import petsc4py
from petsc4py import PETSc

print("dolfinx version:", dolfinx.__version__)
print("Available submodules:", dir(dolfinx))
print("\nmesh module functions:", [x for x in dir(mesh) if not x.startswith('_')])
print("\nfem module functions:", [x for x in dir(fem) if not x.startswith('_')])
