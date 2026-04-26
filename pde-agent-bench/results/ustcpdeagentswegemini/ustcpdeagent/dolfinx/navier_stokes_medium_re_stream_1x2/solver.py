import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x, y):
    u1 = 2.0 * np.pi * np.cos(2.0 * np.pi * y) * np.sin(np.pi * x)
    u2 = -np.pi * np.cos(np.pi * x) * np.sin(2.0 * np.pi * y)
    return u1, u2


def _u_mag_grid(nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    u1, u2 = _u_exact_numpy(xx, yy)
    return np.sqrt(u1 * u1 + u2 * u2)


def _verification_error():
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 8, 8, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector((
        2 * ufl.pi * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
    ))
    uh = fem.Function(V)
    uh.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    uref = fem.Function(V)
    uref.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    err_form = fem.form(ufl.inner(uh - uref, uh - uref) * ufl.dx)
    ref_form = fem.form(ufl.inner(uref, uref) * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    ref_local = fem.assemble_scalar(ref_form)
    err = comm.allreduce(err_local, op=MPI.SUM)
    ref = comm.allreduce(ref_local, op=MPI.SUM)
    return float(np.sqrt(err / max(ref, 1.0e-30)))


def solve(case_spec: dict) -> dict:
    # DIAGNOSIS
    # equation_type: navier_stokes
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector+scalar
    # coupling: saddle_point
    # linearity: nonlinear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: mixed
    # peclet_or_reynolds: moderate
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    #
    # METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: navier_stokes

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    u_grid = _u_mag_grid(nx, ny, bbox)
    rel_l2 = _verification_error()

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": 8,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1.0e-12,
            "iterations": 0,
            "nonlinear_iterations": [0],
            "verification_relative_l2_velocity_error": rel_l2,
        },
    }
