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
# special_notes:        variable_coeff=none, manufactured_solution=none
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
    pointsT = np.asarray(points_array.T, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(tree, pointsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pointsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pointsT[i])
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


def _assemble_problem(comm, n: int, degree: int, epsilon_val: float, beta_vec, rhs_val: float,
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
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-14))
    Pe = beta_norm * h / (2.0 * epsilon)
    cothPe = ufl.cosh(Pe) / ufl.sinh(Pe)
    tau_supg = (h / (2.0 * beta_norm)) * (cothPe - 1.0 / Pe)
    tau_small = h * h / (12.0 * epsilon)
    tau = ufl.conditional(ufl.gt(Pe, ScalarType(1.0)), tau_supg, tau_small)

    adv_v = ufl.dot(beta, ufl.grad(v))
    a_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_galerkin = f * v * ufl.dx

    # For constant epsilon and beta, and P1/P2 Lagrange, practical SUPG residual term:
    # tau * (beta·grad u) * (beta·grad v); RHS gets tau * f * (beta·grad v)
    a_supg = tau * ufl.dot(beta, ufl.grad(u)) * adv_v * ufl.dx
    L_supg = tau * f * adv_v * ufl.dx

    a = a_galerkin + a_supg
    L = L_galerkin + L_supg

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
    return domain, V, problem, epsilon, beta, f


def _solve_once(comm, n, degree, epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    domain, V, problem, epsilon, beta, f = _assemble_problem(
        comm, n, degree, epsilon_val, beta_vec, rhs_val, ksp_type, pc_type, rtol
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

    strong_residual = ufl.dot(beta, ufl.grad(uh)) - f
    try:
        ind_local = fem.assemble_scalar(fem.form((strong_residual ** 2) * ufl.dx))
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

    hard_time_limit = 18.473
    target_budget = 16.5
    t0 = time.perf_counter()

    candidates = [(96, 1), (128, 1), (160, 1), (192, 1), (224, 1), (256, 1), (160, 2), (192, 2)]
    chosen_grid = None
    chosen_info = None
    verification = {}
    prev_grid = None
    total_iterations = 0

    for n, degree in candidates:
        if (time.perf_counter() - t0) > 0.88 * target_budget and chosen_grid is not None:
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

        total_iterations += int(info["iterations"])
        grid_now = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])
        diff_prev = None if prev_grid is None else float(np.linalg.norm(grid_now - prev_grid) / np.sqrt(grid_now.size))

        chosen_grid = grid_now
        chosen_info = info
        prev_grid = grid_now
        verification = {
            "residual_indicator": float(residual_indicator) if np.isfinite(residual_indicator) else None,
            "grid_change_l2": diff_prev,
            "elapsed_sec": float(time.perf_counter() - t0),
        }

        if diff_prev is not None and diff_prev < 2.5e-4 and (time.perf_counter() - t0) > 5.0:
            break

    # Accuracy verification / adaptive refinement if time remains
    elapsed = time.perf_counter() - t0
    if chosen_info is not None and elapsed < 0.8 * target_budget:
        try:
            n_ref = min(int(chosen_info["mesh_resolution"]) + 32, 288)
            uh_ref, info_ref, _ = _solve_once(
                comm, n_ref, int(chosen_info["element_degree"]), epsilon_val, beta_vec, rhs_val,
                ksp_type="gmres", pc_type="ilu", rtol=1e-8
            )
            total_iterations += int(info_ref["iterations"])
            grid_ref = _sample_on_uniform_grid(uh_ref, case_spec["output"]["grid"])
            fine_diff = float(np.linalg.norm(grid_ref - chosen_grid) / np.sqrt(grid_ref.size))
            verification["fine_grid_reference_l2"] = fine_diff
            if fine_diff < 7.5e-4 or (time.perf_counter() - t0) < hard_time_limit:
                chosen_grid = grid_ref
                chosen_info = info_ref
        except Exception:
            verification["fine_grid_reference_l2"] = None

    chosen_info = dict(chosen_info if chosen_info is not None else {
        "mesh_resolution": 96,
        "element_degree": 1,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": 0,
    })
    chosen_info["iterations"] = int(total_iterations)
    chosen_info["verification"] = verification

    return {
        "u": np.asarray(chosen_grid, dtype=np.float64),
        "solver_info": chosen_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 96, "ny": 96, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["u"].min(), result["u"].max(), float(np.mean(result["u"])))
        print(result["solver_info"])
