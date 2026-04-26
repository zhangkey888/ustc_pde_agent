import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _boundary_all(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(idx_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape((ny, nx))
    return None


def _solve_poisson(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f = ufl.exp(-250.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000,
    }
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    iterations = -1
    used_ksp = ksp_type
    used_pc = pc_type
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_{degree}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            ksp = problem.solver
            iterations = int(ksp.getIterationNumber())
            used_ksp = ksp.getType()
            used_pc = ksp.getPC().getType()
        except Exception:
            pass
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_lu_{n}_{degree}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1
        used_ksp = "preonly"
        used_pc = "lu"

    energy = fem.assemble_scalar(fem.form(ufl.inner(kappa * ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    rhs_work = fem.assemble_scalar(fem.form(ufl.inner(f, uh) * ufl.dx))
    energy = comm.allreduce(energy, op=MPI.SUM)
    rhs_work = comm.allreduce(rhs_work, op=MPI.SUM)
    residual_indicator = abs(energy - rhs_work) / max(1.0, abs(rhs_work))

    return domain, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": int(max(0, iterations)),
        "residual_indicator": float(residual_indicator),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])

    target = max(nx, ny)
    if target <= 64:
        n = 80
    elif target <= 128:
        n = 96
    else:
        n = 128
    degree = 2
    coarse_n = max(32, n // 2)

    coarse_domain, coarse_u, coarse_info = _solve_poisson(coarse_n, degree=degree)
    fine_domain, fine_u, fine_info = _solve_poisson(n, degree=degree)

    coarse_grid = _sample_function_on_grid(coarse_domain, coarse_u, grid)
    fine_grid = _sample_function_on_grid(fine_domain, fine_u, grid)

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        diff = np.linalg.norm(fine_grid - coarse_grid) / math.sqrt(fine_grid.size)
        solver_info = {
            "mesh_resolution": fine_info["mesh_resolution"],
            "element_degree": fine_info["element_degree"],
            "ksp_type": fine_info["ksp_type"],
            "pc_type": fine_info["pc_type"],
            "rtol": fine_info["rtol"],
            "iterations": int(coarse_info["iterations"] + fine_info["iterations"]),
            "verification": {
                "coarse_mesh_resolution": int(coarse_n),
                "grid_L2_coarse_fine": float(diff),
                "energy_rhs_relative_mismatch": float(fine_info["residual_indicator"]),
                "wall_time_sec": float(time.perf_counter() - t0),
            },
        }
        return {"u": np.asarray(fine_grid, dtype=np.float64), "solver_info": solver_info}
    return None
