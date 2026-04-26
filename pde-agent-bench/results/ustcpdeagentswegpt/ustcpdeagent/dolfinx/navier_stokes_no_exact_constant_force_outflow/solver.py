import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


# ```DIAGNOSIS
# equation_type:        navier_stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            nonlinear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
# solution_regularity:  smooth
# bc_type:              mixed
# special_notes:        pressure_pinning
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       lu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```


def _default_case_spec(case_spec):
    out = dict(case_spec) if case_spec is not None else {}
    out.setdefault("pde", {})
    out.setdefault("output", {})
    out["output"].setdefault(
        "grid",
        {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]},
    )
    return out


def _build_mixed_space(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _make_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1

    def y0(x):
        return np.isclose(x[1], 0.0)

    def y1(x):
        return np.isclose(x[1], 1.0)

    def x1(x):
        return np.isclose(x[0], 1.0)

    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0

    bcs = []
    for marker in (y0, y1, x1):
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        if len(dofs) > 0:
            bcs.append(fem.dirichletbc(zero_u, dofs, W.sub(0)))

    # Pressure pinning for uniqueness
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    return bcs


def _assemble_picard_problem(msh, W, uk, nu_value, force):
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_value))
    f = fem.Constant(msh, PETSc.ScalarType(force))

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * uk, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    return a, L


def _solve_linear_system(a, L, bcs, prefix, ksp_type="gmres", pc_type="lu", rtol=1e-9):
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "pc_type": pc_type,
        },
    )
    wh = problem.solve()
    return wh


def _compute_diagnostics(msh, wh, nu_value, force):
    W = wh.function_space
    wtest = ufl.TestFunction(W)
    (v, q) = ufl.split(wtest)
    (u, p) = ufl.split(wh)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_value))
    f = fem.Constant(msh, PETSc.ScalarType(force))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    res_vec = petsc.assemble_vector(fem.form(F))
    res_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    residual_norm = res_vec.norm()

    div_l2_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u), ufl.div(u)) * ufl.dx))
    div_l2 = np.sqrt(msh.comm.allreduce(div_l2_local, op=MPI.SUM))
    return residual_norm, div_l2


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32), :] = vals

    # Gather from all ranks by taking first non-nan contribution
    gathered = msh.comm.gather(values, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & (~np.isnan(arr[:, 0]))
            merged[mask] = arr[mask]
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    mag = msh.comm.bcast(mag, root=0)
    return mag


def solve(case_spec: dict) -> dict:
    case_spec = _default_case_spec(case_spec)
    comm = MPI.COMM_WORLD

    nu_value = 0.3
    force = np.array((1.0, 0.0), dtype=PETSc.RealType)

    # Adaptive accuracy/time trade-off:
    # Since time limit is generous, use a fairly fine mesh directly.
    mesh_resolution = int(case_spec.get("agent_params", {}).get("mesh_resolution", 72))
    degree_u = int(case_spec.get("agent_params", {}).get("degree_u", 2))
    degree_p = int(case_spec.get("agent_params", {}).get("degree_p", 1))
    picard_tol = float(case_spec.get("agent_params", {}).get("newton_rtol", 1e-9))
    picard_max_it = int(case_spec.get("agent_params", {}).get("newton_max_it", 30))

    t0 = time.perf_counter()
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_mixed_space(msh, degree_u=degree_u, degree_p=degree_p)
    bcs = _make_bcs(msh, W, V, Q)

    wk = fem.Function(W)
    wk.x.array[:] = 0.0
    wk.x.scatter_forward()

    nonlinear_iterations = []
    total_iterations = 0

    prev_u_norm = None
    for it in range(picard_max_it):
        uk, _ = wk.split()
        a, L = _assemble_picard_problem(msh, W, uk, nu_value, force)

        try:
            wh = _solve_linear_system(a, L, bcs, prefix=f"ns_picard_{it}_", ksp_type="gmres", pc_type="lu", rtol=1e-9)
            used_ksp = "gmres"
            used_pc = "lu"
        except Exception:
            wh = _solve_linear_system(a, L, bcs, prefix=f"ns_picard_fb_{it}_", ksp_type="preonly", pc_type="lu", rtol=1e-12)
            used_ksp = "preonly"
            used_pc = "lu"

        diff = wh.x.array - wk.x.array
        diff_norm = np.sqrt(comm.allreduce(np.dot(diff, diff), op=MPI.SUM))
        u_curr, _ = wh.split()
        u_norm_local = np.dot(u_curr.x.array, u_curr.x.array)
        u_norm = np.sqrt(comm.allreduce(u_norm_local, op=MPI.SUM))
        rel = diff_norm / max(u_norm, 1e-14)

        wk.x.array[:] = wh.x.array
        wk.x.scatter_forward()

        nonlinear_iterations.append(it + 1)
        total_iterations += 1

        if rel < picard_tol:
            break
        if prev_u_norm is not None and abs(u_norm - prev_u_norm) < 1e-13:
            break
        prev_u_norm = u_norm

    u_sol, p_sol = wk.split()
    u_out = u_sol.collapse()

    residual_norm, div_l2 = _compute_diagnostics(msh, wk, nu_value, force)

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    u_grid = _sample_velocity_magnitude(u_out, msh, nx, ny, bbox)

    elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": 1e-9,
        "iterations": int(total_iterations),
        "nonlinear_iterations": [int(total_iterations)],
        "wall_time_sec": float(elapsed),
        "verification": {
            "residual_l2": float(residual_norm),
            "divergence_l2": float(div_l2),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
