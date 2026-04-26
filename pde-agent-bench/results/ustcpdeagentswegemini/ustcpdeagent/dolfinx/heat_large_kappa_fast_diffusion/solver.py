import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _get_time_data(case_spec):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.08))
    dt = float(pde.get("dt", 0.004))
    scheme = str(pde.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _get_kappa(case_spec):
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    return float(coeffs.get("kappa", case_spec.get("kappa", 5.0)))


def _choose_discretization(case_spec):
    time_limit = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 34.734)))
    dt_suggested = _get_time_data(case_spec)[2]
    if time_limit >= 30.0:
        mesh_n = 96
        degree = 2
        dt = min(dt_suggested, 0.001)
    elif time_limit >= 15.0:
        mesh_n = 80
        degree = 2
        dt = min(dt_suggested, 0.0015)
    else:
        mesh_n = 64
        degree = 2
        dt = min(dt_suggested, 0.002)
    return mesh_n, degree, dt


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.allgather(local_vals)
    vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        vals[mask] = arr[mask]

    return vals.reshape(ny, nx)


def _solve_once(case_spec, mesh_n, degree, dt_used, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    t0, t_end, _, scheme = _get_time_data(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"
    kappa = _get_kappa(case_spec)

    n_steps = max(1, int(round((t_end - t0) / dt_used)))
    dt_used = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(COMM, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    def u_exact_expr(t):
        return ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])

    def f_expr(t):
        ue = u_exact_expr(t)
        return -ue + kappa * (((2 * pi) ** 2) + (pi ** 2)) * ue

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_expr(t0), V.element.interpolation_points))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    t_const = fem.Constant(domain, ScalarType(t0 + dt_used))
    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_expr(t_const), V.element.interpolation_points))
    bc = fem.dirichletbc(uD, bdofs)

    dt_c = fem.Constant(domain, ScalarType(dt_used))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr(t_const) * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    total_iterations = 0

    for n in range(n_steps):
        t_now = t0 + (n + 1) * dt_used
        t_const.value = ScalarType(t_now)
        uD.interpolate(fem.Expression(u_exact_expr(t_const), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its is not None and its >= 0:
            total_iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_ex_T = fem.Function(V)
    u_ex_T.interpolate(fem.Expression(u_exact_expr(t_end), V.element.interpolation_points))
    err = fem.form((uh - u_ex_T) ** 2 * ufl.dx)
    norm = fem.form((u_ex_T) ** 2 * ufl.dx)
    l2_error = math.sqrt(domain.comm.allreduce(fem.assemble_scalar(err), op=MPI.SUM))
    l2_norm = math.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm), op=MPI.SUM))
    rel_l2_error = l2_error / (l2_norm + 1e-16)

    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u_exact_expr(t0), V.element.interpolation_points))

    return {
        "u_final": uh,
        "u_initial": u_init,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt_used,
        "n_steps": n_steps,
        "time_scheme": scheme,
        "l2_error": l2_error,
        "relative_l2_error": rel_l2_error,
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    mesh_n, degree, dt = _choose_discretization(case_spec)
    time_limit = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 34.734)))

    attempts = []
    candidates = [(mesh_n, degree, dt)]
    if time_limit >= 25.0:
        candidates.append((112, degree, min(dt, 0.0008)))

    result = None
    for mn, deg, dt_use in candidates:
        result = _solve_once(case_spec, mn, deg, dt_use)
        attempts.append(
            {
                "mesh_resolution": mn,
                "element_degree": deg,
                "dt": dt_use,
                "l2_error": result["l2_error"],
            }
        )
        elapsed = time.perf_counter() - start
        if elapsed > 0.5 * time_limit:
            break

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(result["u_final"], grid_spec)
    u_initial = _sample_on_grid(result["u_initial"], grid_spec)

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["element_degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": str(result["time_scheme"]),
        "l2_error": float(result["l2_error"]),
        "relative_l2_error": float(result["relative_l2_error"]),
        "attempts": attempts,
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.asarray(u_initial, dtype=np.float64),
        "solver_info": solver_info,
    }
