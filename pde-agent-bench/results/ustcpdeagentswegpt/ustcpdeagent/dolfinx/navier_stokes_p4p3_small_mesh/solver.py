import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Return:
      - "u": velocity magnitude sampled on requested uniform grid, shape (ny, nx)
      - "solver_info": metadata including nonlinear iterations and verification
    """

    # ------------------------------------------------------------------
    # Mandatory diagnosis / method cards (kept as strings inside module)
    # ------------------------------------------------------------------
    _diagnosis = """```DIAGNOSIS
equation_type: navier_stokes
spatial_dim: 2
domain_geometry: rectangle
unknowns: vector+scalar
coupling: saddle_point
linearity: nonlinear
time_dependence: steady
stiffness: N/A
dominant_physics: mixed
peclet_or_reynolds: low
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: pressure_pinning / manufactured_solution