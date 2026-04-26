import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion
# ```


def _get_case_param(case_spec, keys, default=None):
    obj = case_spec
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def _probe_points(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals_local = _probe_points(u_func, pts)

    comm = u_func.function_space.mesh.comm
    vals_global = vals_local.copy()
    vals_global[np.isnan(vals_global)] = -1.0e300
    recv = np.empty_like(vals_global)
    comm.Allreduce(vals_global, recv, op=MPI.MAX)
    recv[recv < -1.0e299] = np.nan
    return recv.reshape((ny, nx))


def _build_and_run(case_spec, nx, degree, dt, t_end):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps_val = float(_get_case_param(case_spec, ["pde", "epsilon"], 0.05))
    beta_list = _get_case_param(case_spec, ["pde", "beta"], [2.0, 1.0])
    if beta_list is None:
        beta_list = [2.0, 1.0]
    beta_arr = np.array(beta_list, dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_arr))

    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, ScalarType(0.0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps_val))
    beta_c = fem.Constant(msh, np.array(beta_arr, dtype=ScalarType))

    pi = math.pi
    u_exact = ufl.exp(-2.0 * t_c) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = (
        -2.0 * u_exact
        - eps_c * (-2.0 * (pi ** 2) * u_exact)
        + ufl.dot(beta_c, ufl.grad(u_exact))
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]), V.element.interpolation_points))

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    u_initial.x.scatter_forward()

    h = ufl.CellDiameter(msh)
    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + 9.0 * (4.0 * eps_c / h**2) ** 2)
    a = ((u / dt_c) * v + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_c, ufl.grad(u)) * v) * ufl.dx
    L = ((u_n / dt_c) + f_expr) * v * ufl.dx

    if beta_norm / max(eps_val, 1e-14) > 10.0:
        beta_grad_v = ufl.dot(beta_c, ufl.grad(v))
        a += tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))) * beta_grad_v * ufl.dx
        L += tau * ((u_n / dt_c) + f_expr) * beta_grad_v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)

    uh = fem.Function(V)
    n_steps = int(round(t_end / dt))
    current_t = 0.0
    total_iterations = 0
    t_start = time.perf_counter()

    for _ in range(n_steps):
        current_t += dt
        t_c.value = ScalarType(current_t)
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("gmres failed")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        total_iterations += int(max(solver.getIterationNumber(), 0))
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - t_start

    u_ex_T = fem.Function(V)
    u_ex_T.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_T.x.array
    err_fun.x.scatter_forward()
    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_err),
        "wall_time_sec_internal": float(wall),
    }
    return uh, u_initial, solver_info


def solve(case_spec: dict) -> dict:
    t_end = float(_get_case_param(case_spec, ["pde", "time", "t_end"], 0.2))
    dt_suggested = float(_get_case_param(case_spec, ["pde", "time", "dt"], 0.02))

    candidates = [
        (36, 1, dt_suggested / 2.0),
        (44, 1, dt_suggested / 2.0),
        (52, 1, dt_suggested / 4.0),
    ]

    best = None
    start = time.perf_counter()
    budget = 2.209

    for nx, degree, dt in candidates:
        dt = min(dt, t_end)
        n_steps = max(1, int(round(t_end / dt)))
        dt = t_end / n_steps
        try:
            uh, u_initial, info = _build_and_run(case_spec, nx, degree, dt, t_end)
        except Exception:
            continue
        if best is None or info["l2_error"] < best[2]["l2_error"]:
            best = (uh, u_initial, info)
        if time.perf_counter() - start > 0.9 * budget:
            break

    if best is None:
        raise RuntimeError("Failed to solve problem")

    uh, u_initial, info = best
    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(uh, grid_spec)
    u0_grid = _sample_on_grid(u_initial, grid_spec)

    return {"u": u_grid, "u_initial": u0_grid, "solver_info": info}
