import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _build_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _make_velocity_bc_function(V, value):
    f = fem.Function(V)
    f.interpolate(
        lambda x: np.vstack(
            [
                np.full(x.shape[1], value[0], dtype=np.float64),
                np.full(x.shape[1], value[1], dtype=np.float64),
            ]
        )
    )
    return f


def _locate_boundary_facets(msh):
    fdim = msh.topology.dim - 1
    return {
        "x0": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0)),
        "x1": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0)),
        "y0": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0)),
        "y1": mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0)),
    }


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    facets = _locate_boundary_facets(msh)
    bcs = []

    u_noslip = _make_velocity_bc_function(V, (0.0, 0.0))
    u_lid = _make_velocity_bc_function(V, (1.0, 0.0))

    for key in ("x0", "x1", "y0"):
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets[key])
        bcs.append(fem.dirichletbc(u_noslip, dofs, W.sub(0)))

    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets["y1"])
    bcs.append(fem.dirichletbc(u_lid, dofs_top, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    return bcs


def _solve_stokes_once(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, degree_u=2, degree_p=1)
    bcs = _build_bcs(msh, W, V, Q)

    nu = ScalarType(0.2)
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=ScalarType))

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    opts_try = [
        {"ksp_type": "minres", "pc_type": "lu"},
        {"ksp_type": "gmres", "pc_type": "lu"},
        {"ksp_type": "preonly", "pc_type": "lu"},
    ]

    last_err = None
    for i, opts in enumerate(opts_try):
        prefix = f"stokes_{n}_{i}_"
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix=prefix,
                petsc_options={
                    **opts,
                    "ksp_rtol": 1.0e-10,
                    "ksp_atol": 1.0e-12,
                    "ksp_max_it": 10000,
                },
            )
            wh = problem.solve()
            wh.x.scatter_forward()

            ksp = problem.solver
            its = int(ksp.getIterationNumber())

            uh = wh.sub(0).collapse()
            ph = wh.sub(1).collapse()

            div_l2 = np.sqrt(
                comm.allreduce(
                    fem.assemble_scalar(
                        fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)
                    ),
                    op=MPI.SUM,
                )
            )

            return {
                "mesh": msh,
                "u": uh,
                "p": ph,
                "iterations": its,
                "ksp_type": opts["ksp_type"],
                "pc_type": opts["pc_type"],
                "rtol": 1.0e-10,
                "mesh_resolution": n,
                "element_degree": 2,
                "verification": {"divergence_l2": float(div_l2)},
            }
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All solver options failed for mesh n={n}: {last_err}")


def _sample_velocity_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((nx * ny, 2), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idxs, dtype=np.int32), :] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full((nx * ny, 2), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged[np.isnan(merged[:, 0]), :] = 0.0
        return np.linalg.norm(merged, axis=1).reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _boundary_verification(u_func):
    msh = u_func.function_space.mesh
    comm = msh.comm
    pts = np.array(
        [
            [0.5, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local = np.full((4, 2), np.nan, dtype=np.float64)
    eval_points = []
    cells = []
    ids = []
    for i in range(4):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if eval_points:
        vals = u_func.eval(np.array(eval_points, dtype=np.float64), np.array(cells, dtype=np.int32))
        local[np.array(ids, dtype=np.int32)] = vals

    gathered = comm.gather(local, root=0)
    if comm.rank == 0:
        merged = np.full((4, 2), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        targets = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        errs = np.linalg.norm(merged - targets, axis=1)
        return {
            "bc_pointwise_max_error": float(np.nanmax(errs)),
            "bc_pointwise_mean_error": float(np.nanmean(errs)),
        }
    return {}


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    output_grid = case_spec["output"]["grid"]

    chosen_levels = [48, 64, 80]
    best = None
    for n in chosen_levels:
        best = _solve_stokes_once(n)
        if time.time() - t0 > 120.0:
            break

    u_grid = _sample_velocity_magnitude(best["u"], output_grid)
    verification = dict(best["verification"])
    verification.update(_boundary_verification(best["u"]))

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification": verification,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
