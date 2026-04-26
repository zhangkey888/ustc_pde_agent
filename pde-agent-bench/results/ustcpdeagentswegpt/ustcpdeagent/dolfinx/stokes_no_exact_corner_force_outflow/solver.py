import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


DIAGNOSIS_CARD = r"""```DIAGNOSIS
equation_type: stokes
spatial_dim: 2
domain_geometry: rectangle
unknowns: vector+scalar
coupling: saddle_point
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: low
solution_regularity: smooth
bc_type: mixed
special_notes: pressure_pinning, variable_force, none