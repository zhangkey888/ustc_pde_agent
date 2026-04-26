import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type:        biharmonic
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar+scalar
# coupling:             sequential
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       hypre
# special_treatment:    problem_splitting
# pde_skill:            none
# ```

ScalarType = PETSc.ScalarType


def _probe_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    local_vals = _probe_function(u_func, pts)
    comm = u_func.function_space.mesh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None
    return comm.bcast(out, root=0)


def _exact_solution_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    lam = (10.0 * pi) ** 2 + (8.0 * pi) ** 2
    return (1.0 / lam**2) * ufl.sin(10.0 * pi * x[0]) * ufl.sin(8.0 * pi * x[1])


def _solve_once(n, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    f_expr = ufl.sin(10.0 * pi * x[0]) * ufl.sin(8.0 * pi * x[1])

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    bc = fem.dirichletbc(zero, boundary_dofs)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    problem_w = petsc.LinearProblem(
        a, ufl.inner(f_expr, v) * ufl.dx, bcs=[bc],
        petsc_options=opts, petsc_options_prefix=f"bihar_w_{n}_"
    )
    w_h = problem_w.solve()
    w_h.x.scatter_forward()

    problem_u = petsc.LinearProblem(
        a, ufl.inner(w_h, v) * ufl.dx, bcs=[bc],
        petsc_options=opts, petsc_options_prefix=f"bihar_u_{n}_"
    )
    u_h = problem_u.solve()
    u_h.x.scatter_forward()

    u_exact = _exact_solution_expr(domain)
    err_L2_local = fem.assemble_scalar(fem.form((u_h - u_exact) ** 2 * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form(u_exact ** 2 * ufl.dx))
    err_H1_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_h - u_exact), ufl.grad(u_h - u_exact)) * ufl.dx))

    err_L2 = np.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = np.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    err_rel = err_L2 / max(norm_L2, 1e-16)
    err_H1 = np.sqrt(comm.allreduce(err_H1_local, op=MPI.SUM))

    return domain, u_h, {
        "l2_error": float(err_L2),
        "relative_l2_error": float(err_rel),
        "h1_semi_error": float(err_H1),
        "iterations": 0,
    }


def solve(case_spec: dict) -> dict:
    solver_opts = case_spec.get("solver_options", {})
    degree = int(solver_opts.get("element_degree", 2))
    ksp_type = str(solver_opts.get("ksp_type", "cg"))
    pc_type = str(solver_opts.get("pc_type", "hypre"))
    rtol = float(solver_opts.get("rtol", 1.0e-10))

    user_n = solver_opts.get("mesh_resolution", None)
    candidates = [int(user_n)] if user_n is not None else [48, 64, 80, 96]

    best = None
    start = time.perf_counter()
    time_budget = 9.5

    for n in candidates:
        domain, u_h, verification = _solve_once(n, degree, ksp_type, pc_type, rtol)
        elapsed = time.perf_counter() - start
        best = (n, domain, u_h, verification, elapsed)
        if user_n is not None:
            break
        if elapsed > time_budget or verification["relative_l2_error"] < 1.0e-3:
            break

    n, domain, u_h, verification, elapsed = best
    u_grid = _sample_on_uniform_grid(u_h, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": float(rtol),
            "iterations": int(verification["iterations"]),
            "verification": verification,
            "wall_time": float(elapsed),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", out["solver_info"]["verification"]["l2_error"])
        print("REL_L2_ERROR:", out["solver_info"]["verification"]["relative_l2_error"])
        print("WALL_TIME:", wall)
        print("GRID_SHAPE:", out["u"].shape)
        print("SOLVER_INFO:", out["solver_info"])
