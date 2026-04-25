import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   high
# solution_regularity:  boundary_layer
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P1
# stabilization:        supg
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion
# ```


ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict) -> dict:
    case = dict(case_spec) if case_spec is not None else {}
    case.setdefault("output", {})
    case["output"].setdefault("grid", {})
    case["output"]["grid"].setdefault("nx", 128)
    case["output"]["grid"].setdefault("ny", 128)
    case["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    case.setdefault("pde", {})
    return case


def _probe_points(u_func: fem.Function, points_array: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)

    ptsT = np.asarray(points_array.T, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(tree, ptsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, ptsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(ptsT[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)
        values[np.array(eval_map, dtype=np.int32)] = vals[:, 0]
    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    local_vals = _probe_points(u_func, points)

    comm = u_func.function_space.mesh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & (~np.isnan(arr))
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        out = merged.reshape(ny, nx)
    else:
        out = None

    return comm.bcast(out, root=0)


def _solve_once(comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    epsilon = fem.Constant(domain, ScalarType(epsilon_val))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=np.float64))
    f = fem.Constant(domain, ScalarType(rhs_val))

    def g_fun(X):
        return np.sin(np.pi * X[0])

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(g_fun)
    bc = fem.dirichletbc(u_bc, dofs)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-16))
    PeK = beta_norm * h / (2.0 * epsilon)

    tau = ufl.conditional(
        ufl.gt(PeK, ScalarType(1.0)),
        h / (2.0 * beta_norm),
        h * h / (12.0 * epsilon),
    )

    strong_operator_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))

    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * strong_operator_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    L = (
        f * v * ufl.dx
        + tau * f * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"convdiff_{n}_{degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 5000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    solver = problem.solver
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
    }

    residual_indicator_form = fem.form(((ufl.dot(beta, ufl.grad(uh)) - f) ** 2) * ufl.dx)
    try:
        res_local = fem.assemble_scalar(residual_indicator_form)
        residual_indicator = math.sqrt(domain.comm.allreduce(res_local, op=MPI.SUM))
    except Exception:
        residual_indicator = float("nan")

    return domain, uh, info, residual_indicator


def solve(case_spec: dict) -> dict:
    case_spec = _make_case_defaults(case_spec)
    comm = MPI.COMM_WORLD

    epsilon_val = 0.005
    beta_vec = [12.0, 0.0]
    rhs_val = 1.0

    time_budget = 82.886
    t0 = time.perf_counter()

    candidates = [(96, 1), (128, 1), (160, 1), (192, 1), (224, 1)]

    chosen = None
    prev_grid = None

    for n, degree in candidates:
        try:
            domain, uh, info, residual_indicator = _solve_once(
                comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8
            )
        except Exception:
            domain, uh, info, residual_indicator = _solve_once(
                comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="preonly", pc_type="lu", rtol=1e-10
            )

        grid_now = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])
        elapsed = time.perf_counter() - t0
        diff_prev = None if prev_grid is None else float(
            np.linalg.norm(grid_now - prev_grid) / np.sqrt(grid_now.size)
        )

        verification = {
            "residual_indicator": float(residual_indicator) if np.isfinite(residual_indicator) else None,
            "grid_change_l2": diff_prev,
            "wall_time_sec_estimate": float(elapsed),
        }

        chosen = (uh, info, grid_now, verification)
        prev_grid = grid_now

        remaining = time_budget - elapsed
        if remaining < 18.0:
            break
        if diff_prev is not None and diff_prev < 8e-4 and elapsed > 8.0:
            break

    uh, info, u_grid, verification = chosen

    elapsed = time.perf_counter() - t0
    if elapsed < 0.6 * time_budget:
        try:
            n_ref = min(info["mesh_resolution"] + 32, 256)
            _, uh_ref, _, _ = _solve_once(
                comm, n_ref, info["element_degree"], epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8
            )
            u_ref_grid = _sample_on_uniform_grid(uh_ref, case_spec["output"]["grid"])
            verification["fine_grid_reference_l2"] = float(
                np.linalg.norm(u_ref_grid - u_grid) / np.sqrt(u_grid.size)
            )
            if verification["fine_grid_reference_l2"] < 3e-4:
                u_grid = u_ref_grid
                info["mesh_resolution"] = int(n_ref)
        except Exception:
            verification["fine_grid_reference_l2"] = None

    solver_info = dict(info)
    solver_info["verification"] = verification

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 96, "ny": 96, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
