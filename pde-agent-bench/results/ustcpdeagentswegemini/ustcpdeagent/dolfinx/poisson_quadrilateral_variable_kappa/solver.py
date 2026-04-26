import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            raise RuntimeError("Failed to evaluate solution at some output grid points.")
        return final.reshape((ny, nx))
    return None


def _compute_errors(domain, uh, u_exact_expr):
    Ve = fem.functionspace(domain, ("Lagrange", max(uh.function_space.element.basix_element.degree, 2)))
    u_exact = fem.Function(Ve)
    u_exact.interpolate(fem.Expression(u_exact_expr, Ve.element.interpolation_points))
    uh_high = fem.Function(Ve)
    uh_high.interpolate(uh)

    err_L2_local = fem.assemble_scalar(fem.form((uh_high - u_exact) ** 2 * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form(u_exact ** 2 * ufl.dx))
    err_H1_local = fem.assemble_scalar(
        fem.form(((uh_high - u_exact) ** 2 + ufl.inner(ufl.grad(uh_high - u_exact), ufl.grad(uh_high - u_exact))) * ufl.dx)
    )

    err_L2 = math.sqrt(domain.comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = math.sqrt(domain.comm.allreduce(norm_L2_local, op=MPI.SUM))
    err_H1 = math.sqrt(domain.comm.allreduce(err_H1_local, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2
    return {"L2_error": err_L2, "relative_L2_error": rel_L2, "H1_error": err_H1}


def _solve_once(mesh_resolution, element_degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    u_exact_expr = ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    kappa_expr = 1.0 + 0.5 * ufl.cos(2.0 * pi * x[0]) * ufl.cos(2.0 * pi * x[1])
    f_expr = -ufl.div(kappa_expr * ufl.grad(u_exact_expr))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, bdofs)

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
            "ksp_max_it": 5000,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    actual_ksp = ksp.getType()
    actual_pc = ksp.getPC().getType()

    errors = _compute_errors(domain, uh, u_exact_expr)

    return {
        "domain": domain,
        "uh": uh,
        "u_exact_expr": u_exact_expr,
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "rtol": rtol,
        "iterations": its,
        "solve_time": solve_time,
        "errors": errors,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    time_limit = 2.918
    configs = [
        (28, 1),
        (40, 1),
        (56, 1),
        (32, 2),
        (40, 2),
        (48, 2),
    ]

    best = None
    wall0 = time.perf_counter()

    for mesh_resolution, element_degree in configs:
        elapsed = time.perf_counter() - wall0
        if elapsed > 0.9 * time_limit:
            break
        try:
            result = _solve_once(mesh_resolution, element_degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            result = _solve_once(mesh_resolution, element_degree, ksp_type="preonly", pc_type="lu", rtol=1e-10)

        if best is None:
            best = result
        else:
            if result["errors"]["L2_error"] < best["errors"]["L2_error"]:
                best = result

        projected_total = (time.perf_counter() - wall0) * 1.35
        if projected_total > time_limit and best["errors"]["L2_error"] <= 4.13e-3:
            break

    if best is None:
        raise RuntimeError("No solver configuration succeeded.")

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "L2_error": float(best["errors"]["L2_error"]),
        "relative_L2_error": float(best["errors"]["relative_L2_error"]),
        "H1_error": float(best["errors"]["H1_error"]),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
