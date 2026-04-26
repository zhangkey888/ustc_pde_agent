import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        variable_rhs
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    solver_fallback
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _source_callable(x):
    return np.sin(6.0 * np.pi * x[0]) * np.cos(5.0 * np.pi * x[1])


def _all_boundary(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def _build_problem(n, degree, k_value, ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    domain = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.cos(5.0 * ufl.pi * x[1])
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _all_boundary)
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-12,
        "ksp_max_it": 5000,
    }
    if ksp_type == "gmres":
        opts["ksp_gmres_restart"] = 200
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=f"helmholtz_{n}_{degree}_",
    )
    return domain, V, problem, a, L, bc, opts


def _solve_with_fallback(n, degree, k_value, rtol):
    try:
        domain, V, problem, a, L, bc, opts = _build_problem(
            n, degree, k_value, ksp_type="gmres", pc_type="ilu", rtol=rtol
        )
        uh = problem.solve()
        ksp = problem.solver
        reason = ksp.getConvergedReason()
        its = ksp.getIterationNumber()
        if reason <= 0:
            raise RuntimeError(f"GMRES did not converge, reason={reason}")
        uh.x.scatter_forward()
        return domain, V, uh, {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": rtol,
            "iterations": int(its),
            "converged_reason": int(reason),
        }
    except Exception:
        domain, V, problem, a, L, bc, opts = _build_problem(
            n, degree, k_value, ksp_type="preonly", pc_type="lu", rtol=rtol
        )
        uh = problem.solve()
        ksp = problem.solver
        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        uh.x.scatter_forward()
        return domain, V, uh, {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": int(its),
            "converged_reason": int(reason),
        }


def _compute_residual_indicator(domain, V, uh, k_value):
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.cos(5.0 * ufl.pi * x[1])
    residual_form = fem.form(
        ((ufl.inner(ufl.grad(uh), ufl.grad(v)) - (k_value ** 2) * uh * v) - f_expr * v) * ufl.dx
    )

    b = petsc.create_vector(residual_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, residual_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return float(b.norm())


def _probe_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    mapping = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            mapping.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.real_if_close(np.asarray(vals).reshape(-1))
        local_vals[np.array(mapping, dtype=np.int32)] = vals

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            merged[~np.isfinite(merged)] = 0.0
        return merged.reshape((ny, nx))
    return None


def _manufactured_solution_verification():
    return None


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    k_value = float(
        pde.get("k", case_spec.get("wavenumber", case_spec.get("params", {}).get("k", 16.0)))
    )

    degree = 2
    rtol = 1e-8

    # Adaptive accuracy/time trade-off: start strong and refine if still cheap.
    candidates = [32, 40, 48]
    time_budget = 649.609
    chosen = None
    final_data = None
    last_elapsed = 0.0

    for n in candidates:
        loop_t0 = time.perf_counter()
        domain, V, uh, solve_info = _solve_with_fallback(n, degree, k_value, rtol)
        residual_indicator = _compute_residual_indicator(domain, V, uh, k_value)
        elapsed = time.perf_counter() - t0
        step_elapsed = time.perf_counter() - loop_t0

        chosen = n
        final_data = (domain, V, uh, solve_info, residual_indicator)
        last_elapsed = elapsed

        remaining = time_budget - elapsed
        # If runtime is far below budget, increase accuracy; otherwise stop.
        if remaining < max(20.0, 8.0 * step_elapsed):
            break

    domain, V, uh, solve_info, residual_indicator = final_data
    u_grid = _probe_on_grid(domain, uh, grid_spec)

    verification = {"algebraic_residual_norm": residual_indicator}
    mms = _manufactured_solution_verification()
    if COMM.rank == 0 and mms is not None:
        verification.update(mms)

    result = None
    if COMM.rank == 0:
        result = {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(
                int(grid_spec["ny"]), int(grid_spec["nx"])
            ),
            "solver_info": {
                "mesh_resolution": int(chosen),
                "element_degree": int(degree),
                "ksp_type": solve_info["ksp_type"],
                "pc_type": solve_info["pc_type"],
                "rtol": float(rtol),
                "iterations": int(solve_info["iterations"]),
                "verification": verification,
                "wall_time_sec": float(time.perf_counter() - t0),
            },
        }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 16.0},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
