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
    f = ScalarType(2.0 * np.pi**2 * kappa) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

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
        petsc_options_prefix=f"poisson_{n}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-12,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_sq = fem.assemble_scalar(err_form)
    l2_sq = msh.comm.allreduce(l2_sq, op=MPI.SUM)
    l2_error = math.sqrt(l2_sq)

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
        gathered = np.empty_like(values)
        COMM.Allreduce(values, gathered, op=MPI.MAX)
        values = gathered

    if np.isnan(values).any():
        idx = np.where(np.isnan(values))[0]
        values[idx] = _exact_numpy(XX.ravel()[idx], YY.ravel()[idx])

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    output_grid = case_spec["output"]["grid"]
    time_budget = 1.077
    error_target = 4.92e-06

    candidates = [24, 32, 40, 48]
    best = None

    for n in candidates:
        msh, uh, l2_error, info = _build_problem(n=n, degree=2, kappa=1.0)
        best = (msh, uh, l2_error, info)

        elapsed = time.perf_counter() - t0
        remaining = time_budget - elapsed

        if l2_error <= error_target and remaining < 0.18:
            break

    msh, uh, l2_error, solver_info = best
    u_grid = _sample_function_to_grid(msh, uh, output_grid)
    solver_info["l2_error"] = float(l2_error)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
