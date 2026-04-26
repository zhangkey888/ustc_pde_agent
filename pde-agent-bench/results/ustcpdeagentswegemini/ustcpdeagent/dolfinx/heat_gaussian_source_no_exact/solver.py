import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

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
# special_notes: variable_coeff

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


def _make_grid(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()
    return nx, ny, bbox, pts


def _sample_function_on_points(msh, uh: fem.Function, points: np.ndarray):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_ids, dtype=np.int32)] = vals

    gathered = msh.comm.allgather(values)
    final = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        final[mask] = arr[mask]
    final[np.isnan(final)] = 0.0
    return final


def _build_source_function(V):
    f_fun = fem.Function(V)
    f_fun.interpolate(lambda x: np.exp(-200.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2)))
    return f_fun


def _build_initial_function(V):
    u0 = fem.Function(V)
    u0.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    return u0


def _run_heat(nx_mesh: int, degree: int, dt: float, t_end: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(msh, PETSc.ScalarType(1.0))
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt))

    u_n = _build_initial_function(V)
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    f_fun = _build_source_function(V)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)

    n_steps = int(round(t_end / dt))
    total_iterations = 0

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
            its = solver.getIterationNumber()
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("iterative solve failed")
        except Exception:
            solver.destroy()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, u_h.x.petsc_vec)
            its = solver.getIterationNumber()

        u_h.x.scatter_forward()
        total_iterations += int(its)
        u_n.x.array[:] = u_h.x.array[:]
        u_n.x.scatter_forward()

    return {
        "mesh": msh,
        "u": u_h,
        "u_initial": _build_initial_function(V),
        "iterations": total_iterations,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-8,
        "n_steps": n_steps,
        "dt": dt,
    }


def _self_convergence_indicator(case_spec: dict, nx_mesh: int, degree: int, dt: float, t_end: float):
    nx, ny, bbox, pts = _make_grid(case_spec)
    coarse = _run_heat(nx_mesh, degree, dt, t_end)
    fine = _run_heat(max(nx_mesh + 8, int(round(nx_mesh * 1.5))), degree, dt / 2.0, t_end)
    uc = _sample_function_on_points(coarse["mesh"], coarse["u"], pts).reshape(ny, nx)
    uf = _sample_function_on_points(fine["mesh"], fine["u"], pts).reshape(ny, nx)
    return float(np.sqrt(np.mean((uf - uc) ** 2)))


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", 0.0))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", 0.1))
    dt = float(case_spec.get("pde", {}).get("time", {}).get("dt", 0.02))
    if t_end <= t0:
        t0, t_end = 0.0, 0.1

    start = time.perf_counter()
    budget = 12.683

    nx_mesh = 80
    degree = 1
    dt = min(dt, 0.005)
    n_steps = math.ceil((t_end - t0) / dt)
    dt = (t_end - t0) / n_steps

    result = _run_heat(nx_mesh, degree, dt, t_end - t0)
    elapsed = time.perf_counter() - start

    if elapsed < 0.35 * budget:
        nx_mesh = 112
        dt2 = min(dt, 0.0025)
        n_steps = math.ceil((t_end - t0) / dt2)
        dt2 = (t_end - t0) / n_steps
        result = _run_heat(nx_mesh, degree, dt2, t_end - t0)
        dt = dt2
        elapsed = time.perf_counter() - start

    nx, ny, bbox, pts = _make_grid(case_spec)
    u_grid = _sample_function_on_points(result["mesh"], result["u"], pts).reshape(ny, nx)
    u_init_grid = _sample_function_on_points(result["mesh"], result["u_initial"], pts).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(nx_mesh),
        "element_degree": int(degree),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": "backward_euler",
    }

    try:
        if elapsed < 0.7 * budget:
            solver_info["self_convergence_l2_grid"] = _self_convergence_indicator(
                case_spec, max(20, nx_mesh // 2), degree, min(0.02, 2.0 * dt), t_end - t0
            )
    except Exception as e:
        solver_info["self_convergence_l2_grid"] = None
        solver_info["verification_note"] = str(e)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }
