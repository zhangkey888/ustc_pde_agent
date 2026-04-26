import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

DIAGNOSIS = "poisson manufactured-solution variable-coefficient elliptic problem"
METHOD = "FEM P2 CG/Hypre with LU fallback"


def _make_kappa_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))


def _make_exact_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _probe_points(u_func, points_array):
    domain = u_func.function_space.mesh
    pts = np.asarray(points_array, dtype=np.float64)
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals_local = _probe_points(u_func, points)
    vals_local_safe = np.where(np.isnan(vals_local), -1.0e300, vals_local)
    vals_global = np.empty_like(vals_local_safe)
    u_func.function_space.mesh.comm.Allreduce(vals_local_safe, vals_global, op=MPI.MAX)
    vals_global[vals_global < -1.0e250] = 0.0
    return vals_global.reshape((ny, nx))


def _build_and_solve(mesh_resolution, element_degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact_expr = _make_exact_expr(domain)
    kappa_expr = _make_kappa_expr(domain)
    f_expr = -ufl.div(kappa_expr * ufl.grad(u_exact_expr))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    last_exc = None
    for try_ksp, try_pc in [(ksp_type, pc_type), ("preonly", "lu")]:
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options_prefix=f"poisson_{mesh_resolution}_{element_degree}_",
                petsc_options={
                    "ksp_type": try_ksp,
                    "pc_type": try_pc,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1.0e-14,
                },
            )
            uh = problem.solve()
            uh.x.scatter_forward()

            u_ex = fem.Function(V)
            u_ex.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
            e = fem.Function(V)
            e.x.array[:] = uh.x.array - u_ex.x.array
            e.x.scatter_forward()

            l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
            l2_error = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))
            h1_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
            h1_error = math.sqrt(comm.allreduce(h1_sq, op=MPI.SUM))

            try:
                iterations = int(problem.solver.getIterationNumber())
            except Exception:
                iterations = 0

            return {
                "uh": uh,
                "l2_error": l2_error,
                "h1_semi_error": h1_error,
                "iterations": iterations,
                "ksp_type": try_ksp,
                "pc_type": try_pc,
            }
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Linear solve failed: {last_exc}")


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    grid_spec = case_spec["output"]["grid"]

    safety_budget = 1.9 if comm.size == 1 else 1.6
    candidates = [(32, 2), (40, 2), (48, 2), (56, 2), (64, 2)]
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    best = None
    for mesh_resolution, element_degree in candidates:
        if (time.perf_counter() - t0) > safety_budget:
            break
        result = _build_and_solve(mesh_resolution, element_degree, ksp_type, pc_type, rtol)
        result["mesh_resolution"] = mesh_resolution
        result["element_degree"] = element_degree
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result

    if best is None:
        raise RuntimeError("No successful solve completed.")

    u_grid = _sample_on_uniform_grid(best["uh"], grid_spec)
    wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(rtol),
        "iterations": int(best["iterations"]),
        "verification_l2_error": float(best["l2_error"]),
        "verification_h1_semi_error": float(best["h1_semi_error"]),
        "wall_time_sec": float(wall_time),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
