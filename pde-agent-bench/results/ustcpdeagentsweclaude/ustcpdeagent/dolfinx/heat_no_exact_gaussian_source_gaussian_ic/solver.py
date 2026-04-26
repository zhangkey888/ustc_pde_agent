import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

DIAGNOSIS = r"""