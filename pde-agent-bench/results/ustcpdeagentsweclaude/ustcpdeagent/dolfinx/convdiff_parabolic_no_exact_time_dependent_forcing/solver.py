import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
import ufl


comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


"""