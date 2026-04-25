import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _manufactured_ufl(msh, k):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f = (41.0 * ufl.pi**2 - k**2) * u_exact
    return x, u_exact, f


def _build_solver(n, degree, k):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x, u_exact_ufl, f_ufl = _manufactured_ufl(msh, k)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix="helmholtz_",
    )
    return msh, V, uD, u_exact_ufl, problem, opts


def _compute_errors(msh, V, uh, u_exact_ufl):
    comm = msh.comm
    diff = uh - u_exact_ufl
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    l2 = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx))
    h1s = np.sqrt(comm.allreduce(h1s_local, op=MPI.SUM))
    return float(l2), float(h1s)


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idxs = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(idxs, dtype=np.int32)] = np.real(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to sample solution at some output grid points.")
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    k = float(case_spec.get("pde", {}).get("k", 24.0))
    if k <= 0:
        k = 24.0

    time_limit = 10.416
    degree = 2
    n = 128

    if MPI.COMM_WORLD.size > 1:
        n = 96

    msh, V, uD, u_exact_ufl, problem, opts = _build_solver(n, degree, k)
    uh = problem.solve()
    uh.x.scatter_forward()

    elapsed = time.perf_counter() - t0
    if elapsed < 5.0:
        n2 = 160 if MPI.COMM_WORLD.size == 1 else 128
        msh2, V2, uD2, u_exact_ufl2, problem2, opts2 = _build_solver(n2, degree, k)
        uh2 = problem2.solve()
        uh2.x.scatter_forward()
        elapsed2 = time.perf_counter() - t0
        if elapsed2 < time_limit * 0.95:
            msh, V, uD, u_exact_ufl, problem, opts, uh, n = msh2, V2, uD2, u_exact_ufl2, problem2, opts2, uh2, n2

    l2_error, h1_error = _compute_errors(msh, V, uh, u_exact_ufl)

    u_grid = _sample_on_grid(msh, uh, case_spec["output"]["grid"])

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()).lower(),
        "pc_type": str(ksp.getPC().getType()).lower(),
        "rtol": 0.0 if str(ksp.getType()).lower() == "preonly" else float(ksp.getTolerances()[0]),
        "iterations": its,
        "l2_error": l2_error,
        "h1_error": h1_error,
        "wall_time_sec": time.perf_counter() - t0,
    }

    result = {"u": u_grid, "solver_info": solver_info}
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 24.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
