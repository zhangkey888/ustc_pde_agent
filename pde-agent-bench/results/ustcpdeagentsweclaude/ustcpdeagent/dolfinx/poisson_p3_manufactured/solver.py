import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_u_numpy(x):
    return np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_and_solve(n, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = ScalarType(1.0)
    f_expr = -ufl.div(ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _exact_u_numpy(X))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        actual_ksp = ksp.getType()
        actual_pc = ksp.getPC().getType()
    except Exception:
        iterations = 0
        actual_ksp = ksp_type
        actual_pc = pc_type

    return msh, uh, l2_error, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(actual_ksp),
        "pc_type": str(actual_pc),
        "rtol": float(rtol),
        "iterations": int(iterations),
    }


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return (np.sin(2.0 * np.pi * XX) * np.sin(np.pi * YY)).astype(np.float64)


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": solution sampled on requested grid, shape (ny, nx)
    - "solver_info": metadata about the solve
    """
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    degree = 3
    rtol = 1e-12

    # Adaptive accuracy/time trade-off for the strict time budget.
    # Start conservative, then refine if very fast.
    candidates = [
        (18, "preonly", "lu"),
        (24, "preonly", "lu"),
        (32, "preonly", "lu"),
    ]

    target_time = 0.9
    best = None

    for n, ksp_type, pc_type in candidates:
        msh, uh, l2_error, info = _build_and_solve(n, degree, ksp_type, pc_type, rtol)
        elapsed = time.perf_counter() - t0
        best = (msh, uh, l2_error, info, elapsed)
        if elapsed > target_time or l2_error <= 1e-10:
            break

    msh, uh, l2_error, info, elapsed = best

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(msh, uh, grid)

    info["manufactured_l2_error"] = float(l2_error)
    info["wall_time_sec"] = float(elapsed)

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": info}
    return {"u": u_grid, "solver_info": info}
