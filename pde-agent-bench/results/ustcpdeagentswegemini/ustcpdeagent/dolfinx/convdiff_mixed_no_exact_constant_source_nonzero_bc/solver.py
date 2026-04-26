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
# element_or_basis:     Lagrange_P2
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
    if points_on_proc:
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
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(g_fun)
    bc = fem.dirichletbc(u_bc, dofs)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-16))
    PeK = beta_norm * h / (2.0 * epsilon)
    cothPe = ufl.cosh(PeK) / ufl.sinh(PeK)
    tau_stream = (h / (2.0 * beta_norm)) * (cothPe - 1.0 / PeK)
    tau_diff = h * h / (12.0 * epsilon)
    tau = ufl.conditional(ufl.gt(PeK, ScalarType(1.0)), tau_stream, tau_diff)

    Lu = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * ufl.dot(beta, ufl.grad(v)) * ufl.dx
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
            "ksp_max_it": 10000,
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

    strong_residual = -epsilon * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - f
    indicator_form = fem.form((strong_residual ** 2) * ufl.dx)
    try:
        ind_local = fem.assemble_scalar(indicator_form)
        residual_indicator = math.sqrt(domain.comm.allreduce(ind_local, op=MPI.SUM))
    except Exception:
        residual_indicator = float("nan")

    return uh, info, residual_indicator


def solve(case_spec: dict) -> dict:
    case_spec = _make_case_defaults(case_spec)
    comm = MPI.COMM_WORLD

    epsilon_val = 0.005
    beta_vec = [12.0, 0.0]
    rhs_val = 1.0

    time_budget = 23.0
    t0 = time.perf_counter()

    candidates = [(96, 1), (128, 1), (160, 1), (192, 1), (128, 2), (160, 2), (192, 2), (224, 2)]
    chosen_uh = None
    chosen_info = None
    chosen_grid = None
    verification = {}
    prev_grid = None

    for n, degree in candidates:
        elapsed = time.perf_counter() - t0
        if elapsed > 0.92 * time_budget and chosen_uh is not None:
            break

        try:
            uh, info, residual_indicator = _solve_once(
                comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8
            )
        except Exception:
            uh, info, residual_indicator = _solve_once(
                comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="preonly", pc_type="lu", rtol=1e-10
            )

        grid_now = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])
        diff_prev = None if prev_grid is None else float(np.linalg.norm(grid_now - prev_grid) / np.sqrt(grid_now.size))

        verification = {
            "residual_indicator": float(residual_indicator) if np.isfinite(residual_indicator) else None,
            "grid_change_l2": diff_prev,
            "wall_time_sec_estimate": float(time.perf_counter() - t0),
        }

        chosen_uh = uh
        chosen_info = info
        chosen_grid = grid_now
        prev_grid = grid_now

        if diff_prev is not None and diff_prev < 2.0e-4 and (time.perf_counter() - t0) > 6.0:
            break

    elapsed = time.perf_counter() - t0
    if elapsed < 0.9 * time_budget:
        try:
            n_ref = min(int(chosen_info["mesh_resolution"]) + (24 if int(chosen_info["element_degree"]) > 1 else 32), 256)
            uh_ref, info_ref, _ = _solve_once(
                comm, n_ref, int(chosen_info["element_degree"]), epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8
            )
            grid_ref = _sample_on_uniform_grid(uh_ref, case_spec["output"]["grid"])
            verification["fine_grid_reference_l2"] = float(np.linalg.norm(grid_ref - chosen_grid) / np.sqrt(grid_ref.size))
            if verification["fine_grid_reference_l2"] < 5.0e-4:
                chosen_uh = uh_ref
                chosen_info = info_ref
                chosen_grid = grid_ref
        except Exception:
            verification["fine_grid_reference_l2"] = None

    solver_info = dict(chosen_info)
    solver_info["verification"] = verification

    return {
        "u": np.asarray(chosen_grid, dtype=np.float64),
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
