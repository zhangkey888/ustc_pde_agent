# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

from __future__ import annotations

import time
import math
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _get_nested(d: dict, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_case(case_spec: dict):
    pde = case_spec.get("pde", {})
    coeffs = case_spec.get("coefficients", case_spec.get("coeffs", {}))
    time_spec = case_spec.get("time", case_spec.get("pde", {}).get("time", {}))

    t0 = float(_get_nested(case_spec, ["time", "t0"], _get_nested(pde, ["time", "t0"], 0.0)) or 0.0)
    t_end = float(_get_nested(case_spec, ["time", "t_end"], _get_nested(pde, ["time", "t_end"], 0.2)) or 0.2)
    dt_suggested = float(_get_nested(case_spec, ["time", "dt"], _get_nested(pde, ["time", "dt"], 0.02)) or 0.02)
    scheme = _get_nested(case_spec, ["time", "scheme"], _get_nested(pde, ["time", "scheme"], "backward_euler")) or "backward_euler"

    kappa = _get_nested(case_spec, ["coefficients", "kappa"], None)
    if kappa is None:
        kappa = _get_nested(case_spec, ["kappa"], None)
    if kappa is None:
        kappa = 0.8
    kappa = float(kappa)

    out_grid = _get_nested(case_spec, ["output", "grid"], {})
    nx_out = int(out_grid.get("nx", 64))
    ny_out = int(out_grid.get("ny", 64))
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    return t0, t_end, dt_suggested, scheme, kappa, nx_out, ny_out, bbox


def _build_boundary_condition(V, domain):
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    return bc


def _initial_condition(u0_fun: fem.Function):
    u0_fun.interpolate(
        lambda x: np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])
    )


def _sample_on_grid(domain, uh: fem.Function, nx: int, ny: int, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if len(pts_local) > 0:
        vals = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(ids_local), -1)
        values[np.array(ids_local, dtype=np.int32)] = vals[:, 0]

    # Gather ownership in parallel and reduce NaNs away
    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        merged[mask] = arr[mask]

    # For robustness on boundary points, fill any remaining NaN with 0 (Dirichlet boundary / outside tolerance)
    merged = np.nan_to_num(merged, nan=0.0)
    return merged.reshape((ny, nx))


def _run_heat_solver(
    n: int,
    degree: int,
    dt: float,
    t0: float,
    t_end: float,
    kappa: float,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
    rtol: float = 1e-10,
):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u_n = fem.Function(V)
    _initial_condition(u_n)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    bc = _build_boundary_condition(V, domain)
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array.copy()

    t = t0
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    energies = []

    mass_form = fem.form(uh * uh * ufl.dx)
    t_start = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            its = solver.getIterationNumber()
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("Iterative solver did not converge")
        except Exception:
            # Fallback to direct LU if iterative solve fails
            solver.destroy()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setTolerances(rtol=rtol)
            solver.solve(b, uh.x.petsc_vec)
            its = 1

        uh.x.scatter_forward()
        total_iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
        t += dt

        energy_local = fem.assemble_scalar(mass_form)
        energy = comm.allreduce(energy_local, op=MPI.SUM)
        energies.append(float(energy))

    wall = time.perf_counter() - t_start
    return {
        "domain": domain,
        "V": V,
        "u_final": uh,
        "u_initial": u_n,  # overwritten below if needed by caller
        "iterations": total_iterations,
        "n_steps": n_steps,
        "wall_time": wall,
        "energies": energies,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
    }


def _solve_with_selfcheck(case_spec: dict):
    t0, t_end, dt_suggested, scheme, kappa, nx_out, ny_out, bbox = _parse_case(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    time_budget = 21.769
    comm = MPI.COMM_WORLD

    # Start with a reasonably accurate setting, then improve if clearly under budget.
    degree = 1
    candidates = [
        (56, min(dt_suggested, 0.01)),
        (72, min(dt_suggested, 0.008)),
        (88, min(dt_suggested, 0.00625)),
        (104, min(dt_suggested, 0.005)),
    ]

    best = None
    best_metric = np.inf

    for n, dt in candidates:
        # Keep dt exactly dividing total interval
        n_steps = max(1, int(math.ceil((t_end - t0) / dt)))
        dt = (t_end - t0) / n_steps

        # Initial state for output before time stepping
        domain0 = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
        V0 = fem.functionspace(domain0, ("Lagrange", degree))
        u0_func = fem.Function(V0)
        _initial_condition(u0_func)

        result = _run_heat_solver(n, degree, dt, t0, t_end, kappa)
        result["u_initial_function"] = u0_func

        # Accuracy verification via self-convergence at center samples on a coarse probe grid
        probe_nx, probe_ny = 25, 25
        u_grid = _sample_on_grid(result["domain"], result["u_final"], probe_nx, probe_ny, [0, 1, 0, 1])

        ref_n = min(n + 12, n + 24)
        ref_dt_steps = max(2, int(round((t_end - t0) / max(dt / 2.0, 1e-6))))
        ref_dt = (t_end - t0) / ref_dt_steps
        ref = _run_heat_solver(ref_n, degree, ref_dt, t0, t_end, kappa)
        u_ref_grid = _sample_on_grid(ref["domain"], ref["u_final"], probe_nx, probe_ny, [0, 1, 0, 1])

        self_err = float(np.linalg.norm(u_grid - u_ref_grid) / np.sqrt(u_grid.size))

        # Energy should generally remain bounded; use it as a sanity metric
        energy_growth = 0.0
        if len(result["energies"]) >= 2:
            energy_growth = max(0.0, max(np.diff(result["energies"])))

        metric = self_err + 1e-10 * energy_growth

        if metric < best_metric:
            best_metric = metric
            best = dict(result)
            best["mesh_resolution"] = n
            best["element_degree"] = degree
            best["dt"] = dt
            best["scheme"] = scheme
            best["self_convergence_error"] = self_err
            best["u_initial_function"] = u0_func

        elapsed_est = result["wall_time"] + ref["wall_time"]
        if elapsed_est > 0.55 * time_budget:
            break

    assert best is not None

    u_grid = _sample_on_grid(best["domain"], best["u_final"], nx_out, ny_out, bbox)
    u0_grid = _sample_on_grid(best["u_initial_function"].function_space.mesh, best["u_initial_function"], nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": str(best["scheme"]),
        "accuracy_verification": {
            "type": "self_convergence",
            "l2_grid_difference": float(best["self_convergence_error"]),
            "energy_samples": [float(v) for v in best["energies"][: min(5, len(best["energies"]))]],
        },
    }

    return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": final solution sampled on requested uniform grid, shape (ny, nx)
    - "u_initial": initial condition sampled on the same grid
    - "solver_info": metadata about discretization/solver/time stepping
    """
    return _solve_with_selfcheck(case_spec)


if __name__ == "__main__":
    # Minimal smoke test
    case_spec = {
        "coefficients": {"kappa": 0.8},
        "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"}},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
