import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_u_numpy(x):
    return np.exp(3.0 * (x[0] + x[1])) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_exact_ufl(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.exp(3.0 * (x[0] + x[1])) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _build_rhs_ufl(domain, kappa=1.0):
    x = ufl.SpatialCoordinate(domain)
    u_exact = _build_exact_ufl(domain)
    return -ufl.div(kappa * ufl.grad(u_exact))


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    if comm.size > 1:
        recv = np.empty_like(values)
        comm.Allreduce(values, recv, op=MPI.SUM)
        nan_mask = np.isnan(values).astype(np.int32)
        nan_sum = np.empty_like(nan_mask)
        comm.Allreduce(nan_mask, nan_sum, op=MPI.SUM)
        values = recv
        values[nan_sum == comm.size] = np.nan

    return values.reshape((ny, nx))


def _solve_once(mesh_resolution, element_degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(1.0))

    u_exact_ufl = _build_exact_ufl(domain)
    f_ufl = _build_rhs_ufl(domain, kappa)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: _exact_u_numpy(x))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{mesh_resolution}_{element_degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())
    actual_ksp = ksp.getType()
    actual_pc = ksp.getPC().getType()

    err_L2 = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(
                fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
            ),
            op=MPI.SUM,
        )
    )

    return {
        "domain": domain,
        "uh": uh,
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "rtol": float(rtol),
        "iterations": iterations,
        "l2_error": float(err_L2),
        "solve_time": float(solve_time),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]
    time_budget = 2.494

    candidates = [
        {"mesh_resolution": 40, "element_degree": 1, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 56, "element_degree": 1, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 36, "element_degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
        {"mesh_resolution": 48, "element_degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
        {"mesh_resolution": 64, "element_degree": 2, "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-11},
    ]

    chosen = None
    history = []

    for cand in candidates:
        elapsed = time.perf_counter() - t_start
        if elapsed > 0.92 * time_budget:
            break
        try:
            result = _solve_once(**cand)
            history.append(result)
            if result["l2_error"] <= 2.39e-03:
                chosen = result
            projected_total = (time.perf_counter() - t_start) + result["solve_time"]
            if projected_total > 0.92 * time_budget and chosen is not None:
                break
        except Exception:
            try:
                fallback = dict(cand)
                fallback["ksp_type"] = "preonly"
                fallback["pc_type"] = "lu"
                result = _solve_once(**fallback)
                history.append(result)
                if result["l2_error"] <= 2.39e-03:
                    chosen = result
                projected_total = (time.perf_counter() - t_start) + result["solve_time"]
                if projected_total > 0.92 * time_budget and chosen is not None:
                    break
            except Exception:
                continue

    if chosen is None:
        if not history:
            raise RuntimeError("No successful Poisson solve configuration found.")
        chosen = min(history, key=lambda r: r["l2_error"])

    u_grid = _sample_function_on_grid(chosen["domain"], chosen["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(chosen["element_degree"]),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "l2_error": float(chosen["l2_error"]),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
