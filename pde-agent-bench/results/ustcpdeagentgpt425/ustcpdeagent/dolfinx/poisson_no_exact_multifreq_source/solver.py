import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _probe_points_scalar(msh, u_func, points_xyz):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points_xyz)

    local_values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []

    for i in range(points_xyz.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_xyz[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(idx_map), -1)[:, 0]
        local_values[np.array(idx_map, dtype=np.int32)] = vals

    gathered = msh.comm.allgather(local_values)
    global_values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_values) & ~np.isnan(arr)
        global_values[mask] = arr[mask]

    return global_values


def _sample_on_uniform_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    values = _probe_points_scalar(msh, uh, points)

    boundary_mask = (
        np.isclose(points[:, 0], xmin)
        | np.isclose(points[:, 0], xmax)
        | np.isclose(points[:, 1], ymin)
        | np.isclose(points[:, 1], ymax)
    )
    values[boundary_mask & np.isnan(values)] = 0.0
    values = np.nan_to_num(values, nan=0.0)

    return values.reshape((ny, nx))


def _build_rhs_and_exact(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    u_exact = (
        ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) / (((5 * pi) ** 2) + ((3 * pi) ** 2))
        + 0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1]) / (((9 * pi) ** 2) + ((7 * pi) ** 2))
    )
    f_expr = (
        ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1])
        + 0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1])
    )
    return f_expr, u_exact


def _solve_once(comm, n, degree, ksp_type, pc_type, rtol):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr, u_exact_expr = _build_rhs_and_exact(msh)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
        "ksp_max_it": 20000,
    }

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_{degree}_",
            petsc_options=options,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix=f"poisson_fallback_{n}_{degree}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver

    err_sq = fem.assemble_scalar(fem.form((uh - u_exact_expr) * (uh - u_exact_expr) * ufl.dx))
    ref_sq = fem.assemble_scalar(fem.form(u_exact_expr * u_exact_expr * ufl.dx))
    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    ref_sq = comm.allreduce(ref_sq, op=MPI.SUM)

    l2_err = float(np.sqrt(err_sq))
    rel_l2 = float(np.sqrt(err_sq / max(ref_sq, 1e-30)))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "verification_l2_error": l2_err,
        "verification_rel_l2_error": rel_l2,
    }

    return msh, uh, solver_info


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    start = time.perf_counter()

    candidates = [
        (128, 2),
        (160, 2),
        (192, 2),
    ]

    wall_budget = 6.635
    chosen = None
    last_result = None

    for n, degree in candidates:
        t0 = time.perf_counter()
        result = _solve_once(
            comm=comm,
            n=n,
            degree=degree,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1e-10,
        )
        elapsed_local = time.perf_counter() - t0
        elapsed = comm.allreduce(elapsed_local, op=MPI.MAX)
        msh, uh, info = result
        info["solve_wall_time_sec"] = float(elapsed)
        last_result = (msh, uh, info)

        total_so_far = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        remaining = wall_budget - total_so_far

        if info["verification_l2_error"] <= 2.68e-2:
            chosen = (msh, uh, info)
            if remaining < max(0.8, 0.5 * elapsed):
                break

    if chosen is None:
        chosen = last_result

    msh, uh, solver_info = chosen
    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_uniform_grid(msh, uh, grid_spec)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 65,
                "ny": 49,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
