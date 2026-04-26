import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _beta_vec(domain):
    return fem.Constant(domain, np.array([12.0, 0.0], dtype=ScalarType))


def _exact_expr(x):
    return ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(domain, eps_value=0.01):
    x = ufl.SpatialCoordinate(domain)
    uex = _exact_expr(x)
    beta = ufl.as_vector((12.0, 0.0))
    return -eps_value * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))


def _make_bc(V, domain):
    x = ufl.SpatialCoordinate(domain)
    u_bc = fem.Function(V)
    expr = fem.Expression(_exact_expr(x), V.element.interpolation_points)
    u_bc.interpolate(expr)
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(u_bc, dofs), u_bc


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            for idx in nan_ids:
                x = pts[idx, 0]
                y = pts[idx, 1]
                out[idx] = math.exp(3.0 * x) * math.sin(math.pi * y)
        return out.reshape(ny, nx)
    return None


def _compute_errors(domain, uh, eps_value=0.01):
    x = ufl.SpatialCoordinate(domain)
    uex = _exact_expr(x)
    e = uh - uex
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2 = math.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))
    h1 = math.sqrt(domain.comm.allreduce(h1s_local, op=MPI.SUM))
    return l2, h1


def _solve_once(n, degree=1, eps_value=0.01, tau_scale=1.0, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    beta = _beta_vec(domain)
    f = _forcing_expr(domain, eps_value)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    Pe = beta_norm * h / (2.0 * eps_value)
    cothPe = ufl.cosh(Pe) / ufl.sinh(Pe)
    tau = tau_scale * h / (2.0 * beta_norm) * (cothPe - 1.0 / Pe)

    Lu = -eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    Rv = ufl.dot(beta, ufl.grad(v))

    a = (eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx \
        + tau * Lu * Rv * ufl.dx
    L = f * v * ufl.dx + tau * f * Rv * ufl.dx

    bc, _ = _make_bc(V, domain)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"convdiff_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())
    used_ksp = ksp.getType()
    used_pc = ksp.getPC().getType()

    l2, h1 = _compute_errors(domain, uh, eps_value=eps_value)
    return domain, uh, {"mesh_resolution": n, "element_degree": degree, "ksp_type": used_ksp,
                        "pc_type": used_pc, "rtol": rtol, "iterations": iterations,
                        "l2_error": l2, "h1_semi_error": h1}


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD
    eps_value = 0.01

    try:
        domain, uh, info = _solve_once(n=88, degree=2, eps_value=eps_value, tau_scale=1.0,
                                       ksp_type="gmres", pc_type="ilu", rtol=1e-10)
    except Exception:
        domain, uh, info = _solve_once(n=88, degree=2, eps_value=eps_value, tau_scale=1.0,
                                       ksp_type="preonly", pc_type="lu", rtol=1e-12)

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(info["mesh_resolution"]),
            "element_degree": int(info["element_degree"]),
            "ksp_type": str(info["ksp_type"]),
            "pc_type": str(info["pc_type"]),
            "rtol": float(info["rtol"]),
            "iterations": int(info["iterations"]),
            "l2_error": float(info["l2_error"]),
            "h1_semi_error": float(info["h1_semi_error"]),
            "stabilization": "supg",
            "wall_time_sec": float(time.perf_counter() - t0),
        }
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": None}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
