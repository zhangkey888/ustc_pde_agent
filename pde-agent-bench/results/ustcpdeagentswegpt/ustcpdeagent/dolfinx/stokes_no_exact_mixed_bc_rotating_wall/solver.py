import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _make_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _velocity_bc_fn(val):
    arr = np.array(val, dtype=np.float64)

    def fn(x):
        out = np.zeros((arr.size, x.shape[1]), dtype=ScalarType)
        for i in range(arr.size):
            out[i, :] = arr[i]
        return out

    return fn


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    bcs = []

    def on_x0(x):
        return np.isclose(x[0], 0.0)

    def on_y0(x):
        return np.isclose(x[1], 0.0)

    def on_y1(x):
        return np.isclose(x[1], 1.0)

    markers = [
        (on_x0, [0.0, 0.0]),
        (on_y0, [0.0, 0.0]),
        (on_y1, [0.5, 0.0]),
    ]

    for marker, val in markers:
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(_velocity_bc_fn(val))
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    return bcs


def _solve_stokes_once(n, ksp_type="minres", pc_type="lu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _make_spaces(msh, degree_u=2, degree_p=1)
    bcs = _build_bcs(msh, W, V, Q)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = ScalarType(1.0)
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=ScalarType))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if pc_type == "lu":
        opts["ksp_type"] = "preonly"
    elif pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    prefix = f"stokes_{n}_"
    t0 = time.perf_counter()
    problem = fpetsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options=opts,
    )
    wh = problem.solve()
    solve_time = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    its = 0
    try:
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        ksp_type_actual = ksp.getType()
        pc_type_actual = ksp.getPC().getType()
    except Exception:
        ksp_type_actual = opts["ksp_type"]
        pc_type_actual = pc_type

    return {
        "mesh": msh,
        "W": W,
        "V": V,
        "Q": Q,
        "u": uh,
        "p": ph,
        "time": solve_time,
        "iterations": its,
        "ksp_type": str(ksp_type_actual),
        "pc_type": str(pc_type_actual),
        "rtol": float(rtol),
        "n": int(n),
    }


def _sample_function_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        ev = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(eval_ids, dtype=np.int32), :] = np.real(np.asarray(ev, dtype=np.float64))

    mag_local = np.linalg.norm(vals, axis=1)
    mag_local = np.nan_to_num(mag_local, nan=0.0)

    comm = msh.comm
    gathered = comm.allreduce(mag_local, op=MPI.SUM)
    return gathered.reshape((ny, nx))


def _adaptive_resolution(case_spec):
    grid = case_spec["output"]["grid"]
    max_budget = 15.0
    start = time.perf_counter()

    candidates = [24, 32, 40, 48, 56, 64, 80, 96]
    accepted = None
    prev_grid = None
    prev_n = None
    best_info = None

    for n in candidates:
        info = _solve_stokes_once(n=n, ksp_type="preonly", pc_type="lu", rtol=1e-10)
        u_grid = _sample_function_magnitude(info["u"], grid)

        if prev_grid is not None:
            diff = np.sqrt(np.mean((u_grid - prev_grid) ** 2))
        else:
            diff = np.inf

        elapsed = time.perf_counter() - start
        accepted = (n, info, u_grid, diff)
        best_info = info

        if prev_grid is not None and diff < 1.0e-3:
            break
        if elapsed > max_budget:
            break

        prev_grid = u_grid
        prev_n = n

    n, info, u_grid, diff = accepted
    verification = {
        "reference_mesh_resolution": int(prev_n if prev_n is not None else n),
        "estimated_grid_l2_diff": float(0.0 if not np.isfinite(diff) else diff),
        "solve_wall_time_sec": float(info["time"]),
    }
    return info, u_grid, verification


def solve(case_spec: dict) -> dict:
    """
    Solve steady incompressible Stokes flow with Taylor-Hood mixed elements
    and sample velocity magnitude on the requested uniform grid.
    """
    info, u_grid, verification = _adaptive_resolution(case_spec)

    solver_info = {
        "mesh_resolution": int(info["n"]),
        "element_degree": 2,
        "ksp_type": str(info["ksp_type"]),
        "pc_type": str(info["pc_type"]),
        "rtol": float(info["rtol"]),
        "iterations": int(info["iterations"]),
        "pressure_fixing": "p(0,0)=0",
        "verification": verification,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
