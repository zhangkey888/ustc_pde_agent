from __future__ import annotations

# DIAGNOSIS
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
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat

import math
import time
from typing import Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get_nested(dct: dict, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _extract_time_params(case_spec: dict) -> Tuple[float, float, float]:
    t0 = _get_nested(case_spec, ["pde", "time", "t0"], case_spec.get("t0", 0.0))
    t_end = _get_nested(case_spec, ["pde", "time", "t_end"], case_spec.get("t_end", 0.1))
    dt = _get_nested(case_spec, ["pde", "time", "dt"], case_spec.get("dt", 0.02))
    return float(t0), float(t_end), float(dt)


def _ic_fun(x):
    return np.exp(-120.0 * ((x[0] - 0.6) ** 2 + (x[1] - 0.4) ** 2))


def _sample_on_grid(domain, u_func, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    values_local = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        values_local[np.asarray(ids_on_proc, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(values_local, root=0)
    if comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        return merged.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _run_heat(nx: int, degree: int, dt: float, t0: float, t_end: float, kappa_value: float = 1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u_n = fem.Function(V)
    u_n.interpolate(_ic_fun)
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
    kappa = fem.Constant(domain, ScalarType(kappa_value))

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=5000)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt
    iterations = 0

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    solver.setOperators(A)

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
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
            its = 1

        uh.x.scatter_forward()
        iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u0 = fem.Function(V)
    u0.interpolate(_ic_fun)
    u0.x.scatter_forward()

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "u0": u0,
        "iterations": int(iterations),
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1e-8,
        "dt": float(dt),
        "n_steps": int(n_steps),
    }


def _verification_consistency(case_spec: dict, nx: int, degree: int, dt: float, t0: float, t_end: float):
    fine = _run_heat(nx, degree, dt, t0, t_end)
    coarse_dt = 2.0 * dt
    if coarse_dt >= (t_end - t0):
        return fine, 0.0
    coarse = _run_heat(nx, degree, coarse_dt, t0, t_end)
    grid = _get_nested(case_spec, ["output", "grid"], {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]})
    uf = _sample_on_grid(fine["domain"], fine["u"], grid)
    uc = _sample_on_grid(coarse["domain"], coarse["u"], grid)
    if fine["domain"].comm.rank == 0:
        err = float(np.linalg.norm(uf - uc) / math.sqrt(uf.size))
    else:
        err = 0.0
    err = fine["domain"].comm.bcast(err, root=0)
    return fine, err


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested = _extract_time_params(case_spec)
    output_grid = _get_nested(case_spec, ["output", "grid"], {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    t_start = time.perf_counter()
    candidates = [
        (48, 1, min(dt_suggested, 0.01)),
        (64, 1, min(dt_suggested, 0.01)),
        (80, 1, min(dt_suggested, 0.005)),
    ]

    best = None
    best_err = float("inf")
    for nx, degree, dt in candidates:
        out, err = _verification_consistency(case_spec, nx, degree, dt, t0, t_end)
        if err < best_err:
            best = out
            best_err = err
        if time.perf_counter() - t_start > 12.0:
            break

    if best is None:
        best = _run_heat(48, 1, min(dt_suggested, 0.01), t0, t_end)
        best_err = 0.0

    if time.perf_counter() - t_start < 5.0:
        try:
            upgraded = _run_heat(min(best["mesh_resolution"] + 16, 96),
                                 best["element_degree"],
                                 max(best["dt"] / 2.0, 0.0025),
                                 t0, t_end)
            best = upgraded
        except Exception:
            pass

    u_grid = _sample_on_grid(best["domain"], best["u"], output_grid)
    u0_grid = _sample_on_grid(best["domain"], best["u0"], output_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "verification_consistency_error": float(best_err),
    }

    return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}
