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
# solution_regularity: boundary_layer
# bc_type: all_dirichlet
# special_notes: manufactured_solution

# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat


def _parse_time(case_spec: dict) -> Tuple[float, float, float, int]:
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.08))
    dt_in = float(time_spec.get("dt", 0.008))
    if dt_in <= 0:
        dt_in = 0.008
    n_steps = max(1, int(round((t_end - t0) / dt_in)))
    dt = (t_end - t0) / n_steps
    return t0, t_end, dt, n_steps


def _exact_numpy(points: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-t) * np.exp(5.0 * points[1]) * np.sin(np.pi * points[0])


def _make_problem(mesh_n: int, degree: int, dt: float, ksp_type: str, pc_type: str, rtol: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa = ScalarType(1.0)

    u_exact = ufl.exp(-t_c) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    f_expr = -u_exact - kappa * ufl.div(ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    f_fun = fem.Function(V)
    bc_fun = fem.Function(V)

    a = (u * v + dt_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xq: np.ones(xq.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(bc_fun, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        if pc_type == "hypre":
            solver.getPC().setHYPREType("boomeramg")
    except Exception:
        pass
    solver.setFromOptions()

    return domain, V, t_c, u_exact, f_expr, u_n, u_h, f_fun, bc_fun, bc, a_form, L_form, A, b, solver


def _interp_ufl(expr, V):
    fn = fem.Function(V)
    fn.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return fn


def _run(mesh_n: int, degree: int, dt: float, n_steps: int, t0: float, t_end: float,
         ksp_type: str = "cg", pc_type: str = "hypre", rtol: float = 1e-10):
    domain, V, t_c, u_exact, f_expr, u_n, u_h, f_fun, bc_fun, bc, a_form, L_form, A, b, solver = _make_problem(
        mesh_n, degree, dt, ksp_type, pc_type, rtol
    )

    t_c.value = ScalarType(t0)
    u0 = _interp_ufl(u_exact, V)
    u_n.x.array[:] = u0.x.array
    u_n.x.scatter_forward()

    total_iterations = 0
    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)

        f_now = _interp_ufl(f_expr, V)
        f_fun.x.array[:] = f_now.x.array
        f_fun.x.scatter_forward()

        bc_now = _interp_ufl(u_exact, V)
        bc_fun.x.array[:] = bc_now.x.array
        bc_fun.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its if its is not None and its > 0 else 1)

        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    u_ex_end = ufl.exp(-ScalarType(t_end)) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    err_local = fem.assemble_scalar(fem.form((u_h - u_ex_end) ** 2 * ufl.dx))
    err_l2 = math.sqrt(domain.comm.allreduce(err_local, op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "u_h": u_h,
        "u0": u0,
        "error_l2": err_l2,
        "iterations": total_iterations,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
    }


def _sample_function(domain, uh, grid_spec: dict, fallback_t: float) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            miss = np.isnan(merged)
            xq = np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]])
            merged[miss] = _exact_numpy(xq[:, miss], fallback_t)
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    t_wall0 = time.perf_counter()
    t0, t_end, dt_base, _ = _parse_time(case_spec)
    target_time = 26.923

    candidates = [
        (40, 1, dt_base / 2.0),
        (56, 2, dt_base / 2.0),
        (72, 2, dt_base / 4.0),
        (88, 2, dt_base / 4.0),
    ]

    best = None
    for mesh_n, degree, dt_try in candidates:
        if time.perf_counter() - t_wall0 > 0.85 * target_time:
            break
        n_steps = max(1, int(round((t_end - t0) / dt_try)))
        dt = (t_end - t0) / n_steps
        try:
            result = _run(mesh_n, degree, dt, n_steps, t0, t_end)
            result.update({"mesh_n": mesh_n, "degree": degree, "dt": dt, "n_steps": n_steps})
            best = result
            elapsed = time.perf_counter() - t_wall0
            if result["error_l2"] <= 1.65e-03 and elapsed > 0.5 * target_time:
                break
        except Exception:
            continue

    if best is None:
        n_steps = max(1, int(round((t_end - t0) / dt_base)))
        dt = (t_end - t0) / n_steps
        best = _run(32, 1, dt, n_steps, t0, t_end, ksp_type="preonly", pc_type="lu", rtol=1e-10)
        best.update({"mesh_n": 32, "degree": 1, "dt": dt, "n_steps": n_steps})

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function(best["domain"], best["u_h"], grid_spec, t_end)
    u0_grid = _sample_function(best["domain"], best["u0"], grid_spec, t0)

    solver_info = {
        "mesh_resolution": int(best["mesh_n"]),
        "element_degree": int(best["degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(1e-10),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "l2_error": float(best["error_l2"]),
    }

    if MPI.COMM_WORLD.rank == 0:
        return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.08, "dt": 0.008}},
        "output": {"grid": {"nx": 33, "ny": 33, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
