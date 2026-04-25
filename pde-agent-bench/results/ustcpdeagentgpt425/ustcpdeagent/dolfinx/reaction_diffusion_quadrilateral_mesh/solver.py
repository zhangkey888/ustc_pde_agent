import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = "equation_type: reaction_diffusion; spatial_dim: 2; domain_geometry: rectangle; unknowns: scalar; coupling: none; linearity: linear; time_dependence: transient; stiffness: stiff; dominant_physics: mixed; peclet_or_reynolds: N/A; solution_regularity: smooth; bc_type: all_dirichlet; special_notes: manufactured_solution"
METHOD = "spatial_method: fem; element_or_basis: Lagrange_P2; stabilization: none; time_method: backward_euler; nonlinear_solver: none; linear_solver: cg; preconditioner: hypre; special_treatment: none; pde_skill: reaction_diffusion"


def _probe_function(u_func, points_xyz):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points_xyz)
    values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(points_xyz.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_xyz[i])
            cells.append(links[0])
            idx.append(i)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(idx, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    gathered = msh.comm.allgather(values)
    out = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    return out


def _build_and_solve(case_spec, nx, degree, dt, eps=0.1, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.4))
    scheme = pde_time.get("scheme", "backward_euler")
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [nx, nx],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps))

    u_exact = ufl.exp(-t_c) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_exact = ufl.exp(-t_c) * ufl.exp(x[0]) * (1.0 - ufl.pi**2) * ufl.sin(ufl.pi * x[1])
    u_t_exact = -ufl.exp(-t_c) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = u_t_exact - eps_c * lap_u_exact + u_exact

    f_fun = fem.Function(V)
    u_bc = fem.Function(V)
    u_n = fem.Function(V)
    uh = fem.Function(V)

    def update_fields(t):
        t_c.value = ScalarType(t)
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    update_fields(t0)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v / dt_c + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx
    L = (u_n * v / dt_c + f_fun * v) * ufl.dx

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
    total_iterations = 0

    grid = case_spec["output"]["grid"]
    bbox = grid["bbox"]
    gx = np.linspace(bbox[0], bbox[1], int(grid["nx"]))
    gy = np.linspace(bbox[2], bbox[3], int(grid["ny"]))
    XX, YY = np.meshgrid(gx, gy)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size, dtype=np.float64)])
    u_initial_grid = _probe_function(u_n, points).reshape(len(gy), len(gx))

    t = t0
    for _ in range(n_steps):
        t += dt
        update_fields(t)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        try:
            solver.solve(b, uh.x.petsc_vec)
            total_iterations += int(solver.getIterationNumber())
        except Exception:
            fallback = PETSc.KSP().create(comm)
            fallback.setOperators(A)
            fallback.setType("preonly")
            fallback.getPC().setType("lu")
            fallback.solve(b, uh.x.petsc_vec)
            total_iterations += 1
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_grid = _probe_function(uh, points).reshape(len(gy), len(gx))
    update_fields(t_end)
    u_ex_fun = fem.Function(V)
    u_ex_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    err_sq = fem.assemble_scalar(fem.form((uh - u_ex_fun) * (uh - u_ex_fun) * ufl.dx))
    ex_sq = fem.assemble_scalar(fem.form(u_ex_fun * u_ex_fun * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(err_sq, op=MPI.SUM))
    l2_exact = math.sqrt(comm.allreduce(ex_sq, op=MPI.SUM))
    rel_l2_error = l2_error / l2_exact if l2_exact > 0 else l2_error

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": int(nx),
            "element_degree": int(degree),
            "ksp_type": str(solver.getType()),
            "pc_type": str(solver.getPC().getType()),
            "rtol": float(rtol),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": str(scheme),
            "l2_error": float(l2_error),
            "relative_l2_error": float(rel_l2_error),
            "diagnosis": DIAGNOSIS,
            "method": METHOD,
        },
    }


def solve(case_spec: dict) -> dict:
    pde_time = case_spec.get("pde", {}).get("time", {})
    dt_suggested = float(pde_time.get("dt", 0.01))
    candidates = [
        (32, 1, min(dt_suggested, 0.015)),
        (48, 2, min(dt_suggested, 0.01)),
        (64, 2, min(dt_suggested, 0.0075)),
    ]
    wall_budget = 130.846
    chosen = None
    global_start = time.perf_counter()
    for i, (nx, degree, dt) in enumerate(candidates):
        level_start = time.perf_counter()
        result = _build_and_solve(case_spec, nx=nx, degree=degree, dt=dt)
        level_elapsed = time.perf_counter() - level_start
        result["solver_info"]["wall_time_level"] = float(level_elapsed)
        if chosen is None or result["solver_info"]["relative_l2_error"] <= chosen["solver_info"]["relative_l2_error"]:
            chosen = result
        if i < len(candidates) - 1:
            remaining = wall_budget - (time.perf_counter() - global_start)
            if remaining < 1.8 * level_elapsed:
                break
    return chosen


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.4, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0_wall = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0_wall
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {out[solver_info][l2_error]:.12e}")
        print(f"REL_L2_ERROR: {out[solver_info][relative_l2_error]:.12e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(out["u"].shape)
