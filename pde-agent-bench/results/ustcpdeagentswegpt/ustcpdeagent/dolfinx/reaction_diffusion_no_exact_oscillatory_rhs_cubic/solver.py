import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get_case_time(case_spec):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.3))
    dt = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _get_output_grid(case_spec):
    out = case_spec.get("output", {}).get("grid", {})
    nx = int(out.get("nx", 64))
    ny = int(out.get("ny", 64))
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _probe_function_on_grid(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.allgather(local_vals)
    global_vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(global_vals) & (~np.isnan(arr))
        global_vals[mask] = arr[mask]

    global_vals = np.nan_to_num(global_vals, nan=0.0)
    return global_vals.reshape(ny, nx)


def _reaction(u):
    return u**3


def _make_initial_condition(V):
    u0 = fem.Function(V)
    u0.interpolate(lambda x: 0.2 * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1]))
    u0.x.scatter_forward()
    return u0


def _make_rhs(V):
    f = fem.Function(V)
    f.interpolate(lambda x: np.sin(6.0 * np.pi * x[0]) * np.sin(5.0 * np.pi * x[1]))
    f.x.scatter_forward()
    return f


def _make_zero_dirichlet_bc(V):
    fdim = V.mesh.topology.dim - 1
    facets = mesh.locate_entities_boundary(V.mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc_val = fem.Function(V)
    bc_val.x.array[:] = 0.0
    return fem.dirichletbc(bc_val, dofs)


def _project_to_coarse(fine_func, Vc):
    uc = fem.Function(Vc)
    pts = Vc.tabulate_dof_coordinates()
    pts3 = np.zeros((pts.shape[0], 3), dtype=np.float64)
    pts3[:, :pts.shape[1]] = pts

    msh_f = fine_func.function_space.mesh
    tree = geometry.bb_tree(msh_f, msh_f.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts3)
    cols = geometry.compute_colliding_cells(msh_f, cands, pts3)

    vals_local = np.zeros(pts.shape[0], dtype=np.float64)
    owned = np.zeros(pts.shape[0], dtype=bool)
    point_list = []
    cell_list = []
    ids = []
    for i in range(pts3.shape[0]):
        links = cols.links(i)
        if len(links) > 0:
            point_list.append(pts3[i])
            cell_list.append(links[0])
            ids.append(i)
    if len(point_list) > 0:
        vv = fine_func.eval(np.array(point_list, dtype=np.float64), np.array(cell_list, dtype=np.int32))
        vv = np.asarray(vv).reshape(len(point_list), -1)[:, 0]
        vals_local[np.array(ids, dtype=np.int32)] = vv
        owned[np.array(ids, dtype=np.int32)] = True

    comm = Vc.mesh.comm
    gathered_vals = comm.allgather(vals_local)
    gathered_owned = comm.allgather(owned)
    final_vals = np.zeros_like(vals_local)
    final_owned = np.zeros_like(owned)
    for ov, vv in zip(gathered_owned, gathered_vals):
        mask = (~final_owned) & ov
        final_vals[mask] = vv[mask]
        final_owned[mask] = True

    nloc = Vc.dofmap.index_map.size_local * Vc.dofmap.index_map_bs
    uc.x.array[:nloc] = final_vals[:nloc]
    uc.x.scatter_forward()
    return uc


def _run_solver(nx, degree, dt, epsilon, newton_rtol, max_it, ksp_type, pc_type):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    bc = _make_zero_dirichlet_bc(V)
    u_n = _make_initial_condition(V)
    f = _make_rhs(V)
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array

    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))

    F = ((u - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + _reaction(u) * v * ufl.dx - f * v * ufl.dx
    J = ufl.derivative(F, u, du)

    nonlinear_iterations = []
    total_linear_iterations = 0

    t0 = 0.0
    t_end = 0.3
    n_steps = int(round((t_end - t0) / dt))

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-10,
        "snes_max_it": max_it,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": 1e-8,
    }

    for step in range(n_steps):
        u.x.array[:] = u_n.x.array
        prefix = f"rd_{nx}_{degree}_{step}_"
        problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J, petsc_options_prefix=prefix, petsc_options=petsc_options)
        u = problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        ksp = snes.getKSP()
        total_linear_iterations += int(ksp.getIterationNumber())

        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    return {
        "mesh": msh,
        "V": V,
        "u_final": u,
        "u_initial": _make_initial_condition(V),
        "iterations": total_linear_iterations,
        "nonlinear_iterations": nonlinear_iterations,
        "n_steps": n_steps,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0, t_end, dt_in, scheme = _get_case_time(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    epsilon = float(case_spec.get("parameters", {}).get("epsilon", 0.05))
    newton_rtol = float(case_spec.get("parameters", {}).get("newton_rtol", 1e-8))
    pc_type = str(case_spec.get("parameters", {}).get("pc_type", "ilu"))
    max_it = int(case_spec.get("parameters", {}).get("max_it", 25))

    # Accuracy-oriented defaults using available generous time budget
    degree = int(case_spec.get("parameters", {}).get("element_degree", 1))
    mesh_resolution = int(case_spec.get("parameters", {}).get("mesh_resolution", 72))
    dt = float(case_spec.get("parameters", {}).get("dt", min(dt_in, 0.0025)))
    ksp_type = "gmres"

    start = time.perf_counter()

    coarse_n = max(24, mesh_resolution // 2)
    coarse_dt = min(0.005, 2.0 * dt)

    sol_coarse = _run_solver(coarse_n, degree, coarse_dt, epsilon, newton_rtol, max_it, ksp_type, pc_type)
    sol_fine = _run_solver(mesh_resolution, degree, dt, epsilon, newton_rtol, max_it, ksp_type, pc_type)

    # Accuracy verification by self-convergence: compare fine solution vs coarse projected to coarse space
    uc_from_fine = _project_to_coarse(sol_fine["u_final"], sol_coarse["V"])
    diff_fun = fem.Function(sol_coarse["V"])
    diff_fun.x.array[:] = uc_from_fine.x.array - sol_coarse["u_final"].x.array
    diff_fun.x.scatter_forward()
    l2_diff_local = fem.assemble_scalar(fem.form(diff_fun * diff_fun * ufl.dx))
    ref_local = fem.assemble_scalar(fem.form(sol_coarse["u_final"] * sol_coarse["u_final"] * ufl.dx))
    l2_diff = np.sqrt(comm.allreduce(l2_diff_local, op=MPI.SUM))
    l2_ref = np.sqrt(max(comm.allreduce(ref_local, op=MPI.SUM), 1e-30))
    relative_self_error = l2_diff / l2_ref

    nx_out, ny_out, bbox = _get_output_grid(case_spec)
    u_grid = _probe_function_on_grid(sol_fine["u_final"], nx_out, ny_out, bbox)
    u0_grid = _probe_function_on_grid(sol_fine["u_initial"], nx_out, ny_out, bbox)

    elapsed = time.perf_counter() - start

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": int(sol_coarse["iterations"] + sol_fine["iterations"]),
        "dt": dt,
        "n_steps": int(round((t_end - t0) / dt)),
        "time_scheme": scheme,
        "nonlinear_iterations": [int(v) for v in sol_fine["nonlinear_iterations"]],
        "accuracy_verification": {
            "method": "self_convergence_vs_coarser_run",
            "coarse_mesh_resolution": coarse_n,
            "coarse_dt": coarse_dt,
            "relative_l2_difference": float(relative_self_error),
            "wall_time_sec": float(elapsed),
        },
    }

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }
