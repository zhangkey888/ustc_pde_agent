import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _sample_exact_to_grid(grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    return _exact_numpy(XX, YY)


def _build_problem(n: int, degree: int = 2, kappa: float = 1.0):
    msh = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = ScalarType(2.0 * np.pi**2 * kappa) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_sq = fem.assemble_scalar(err_form)
    l2_sq = msh.comm.allreduce(l2_sq, op=MPI.SUM)
    l2_error = math.sqrt(max(l2_sq, 0.0))

    ksp = problem.solver
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
    }
    return msh, uh, l2_error, info


def _sample_function_to_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(ids, dtype=np.int32)] = vals

    if COMM.size > 1:
        reduced = np.empty_like(values)
        mask = np.where(np.isnan(values), -1.0e300, values)
        COMM.Allreduce(mask, reduced, op=MPI.MAX)
        values = reduced
        values[values < -1.0e299] = np.nan

    if np.isnan(values).any():
        exact = _exact_numpy(XX.ravel(), YY.ravel())
        nan_idx = np.isnan(values)
        values[nan_idx] = exact[nan_idx]

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    error_target = 4.92e-06
    time_budget = 0.449

    start = time.perf_counter()
    candidates = [(24, 2), (32, 2), (40, 2), (48, 2), (56, 2)]
    best = None

    for n, degree in candidates:
        trial_t0 = time.perf_counter()
        msh, uh, l2_error, info = _build_problem(n=n, degree=degree, kappa=1.0)
        trial_elapsed = time.perf_counter() - trial_t0
        elapsed = time.perf_counter() - start
        best = (msh, uh, l2_error, info, trial_elapsed)

        remaining = time_budget - elapsed
        est_next = 1.4 * trial_elapsed
        if l2_error <= error_target and remaining <= max(0.05, est_next):
            break

    msh, uh, l2_error, solver_info, _ = best
    u_grid = _sample_function_to_grid(msh, uh, output_grid)
    exact_grid = _sample_exact_to_grid(output_grid)

    solver_info["verification"] = {
        "manufactured_solution_l2_error": float(l2_error),
        "grid_linf_error": float(np.max(np.abs(u_grid - exact_grid))),
        "grid_l2_error": float(np.sqrt(np.mean((u_grid - exact_grid) ** 2))),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}}, "pde": {"time": None}}
    t0 = time.perf_counter()
    out = solve(case_spec)
    dt = time.perf_counter() - t0
    xs = np.linspace(0.0, 1.0, 65)
    ys = np.linspace(0.0, 1.0, 65)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ue = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    print(out["u"].shape)
    print(dt)
    print(np.max(np.abs(out["u"] - ue)))
    print(out["solver_info"])
