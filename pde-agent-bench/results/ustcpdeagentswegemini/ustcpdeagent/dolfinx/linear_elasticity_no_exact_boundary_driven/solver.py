import time
from typing import Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


DIAGNOSIS = r"""```DIAGNOSIS
equation_type: linear_elasticity
spatial_dim: 2
domain_geometry: rectangle
unknowns: vector
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none