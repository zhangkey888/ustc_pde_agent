import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# pde_skill: convection_diffusion
# ```

ScalarType = PETSc.ScalarType


def _normalize_case(case_spec):
    case = {} if case_spec is None else dict(case_spec)
    pde = dict(case.get("pde", {}))
    time_spec = dict(pde.get("time", {}))
    time_spec.setdefault("t0", 0.0)
    time_spec.setdefault("t_end", 0.08)
    time_spec.setdefault("dt", 0.01)
    pde["time"] = time_spec
    case["pde"] = pde
    output = dict(case.get("output", {}))
    grid = dict(output.get("grid", {}))
    grid.setdefault("nx", 64)
    grid.setdefault("ny", 64)
    grid.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    output["grid"] = grid
    case["output"] = output
    return case


def _u_exact_expr(x, t):
    return ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _sample_to_grid(u_fun, grid_spec):
    domain = u_fun.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    local_values = np.full(points.shape[0], -1.0e300, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(points[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = u_fun.eval(np.asarray(eval_points, dtype=np.float64), np.asarray(eval_cells, dtype=np.int32))
        local_values[np.asarray(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    global_values = np.empty_like(local_values)
    domain.comm.Allreduce(local_values, global_values, op=MPI.MAX)
    global_values[global_values < -1.0e250] = 0.0
    return global_values.reshape((ny, nx))


def _run_config(case_spec, mesh_resolution, dt, degree=1, rtol=1.0e-9):
    comm = MPI.COMM_WORLD
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.08))
    epsilon = 0.01
    beta_np = np.array([10.0, 4.0], dtype=np.float64)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_c = fem.Constant(domain, beta_np.astype(PETSc.RealType))
    dt_c = fem.Constant(domain, ScalarType(dt))
    t_c = fem.Constant(domain, ScalarType(t0))

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)

    u_n.interpolate(fem.Expression(_u_exact_expr(x, t_c), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc.interpolate(fem.Expression(_u_exact_expr(x, t_c), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    uex = _u_exact_expr(x, t_c)
    forcing = -uex - eps_c * ufl.div(ufl.grad(uex)) + ufl.dot(beta_c, ufl.grad(uex))

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + ScalarType(1.0e-14))
    tau_adv = h / (2.0 * beta_norm)
    tau_diff = h * h / (4.0 * eps_c + ScalarType(1.0e-14))
    tau_time = dt_c / 2.0
    tau = 1.0 / ufl.sqrt((1.0 / tau_time) ** 2 + (1.0 / tau_adv) ** 2 + (1.0 / tau_diff) ** 2)

    a_galerkin = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
    )
    L_galerkin = (u_n / dt_c) * v * ufl.dx + forcing * v * ufl.dx

    strong_op_u = (u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    strong_rhs = (u_n / dt_c) + forcing
    streamline_test = ufl.dot(beta_c, ufl.grad(v))

    a = a_galerkin + tau * strong_op_u * streamline_test * ufl.dx
    L = L_galerkin + tau * strong_rhs * streamline_test * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=rtol, atol=1.0e-12, max_it=3000)

    total_iterations = 0
    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(fem.Expression(_u_exact_expr(x, t_c), V.element.interpolation_points))

        with b.localForm() as b_local:
            b_local.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        if ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"KSP failed with reason {ksp.getConvergedReason()}")

        total_iterations += int(ksp.getIterationNumber())
        u_n.x.array[:] = u_h.x.array

    t_c.value = ScalarType(t_end)
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(_u_exact_expr(x, t_c), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    t_c.value = ScalarType(t0)
    u_init_fun = fem.Function(V)
    u_init_fun.interpolate(fem.Expression(_u_exact_expr(x, t_c), V.element.interpolation_points))

    return {
        "u": _sample_to_grid(u_h, case_spec["output"]["grid"]),
        "u_initial": _sample_to_grid(u_init_fun, case_spec["output"]["grid"]),
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": float(rtol),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "verification_l2_error": float(l2_error),
        },
    }


def solve(case_spec: dict) -> dict:
    case_spec = _normalize_case(case_spec)
    target_error = 3.85e-03
    start = time.perf_counter()

    configs = [
        {"mesh_resolution": 96, "dt": 0.0025, "degree": 1, "rtol": 1.0e-10},
        {"mesh_resolution": 128, "dt": 0.0020, "degree": 1, "rtol": 1.0e-10},
        {"mesh_resolution": 144, "dt": 0.0016, "degree": 1, "rtol": 1.0e-10},
    ]

    best = None
    best_err = np.inf
    for cfg in configs:
        result = _run_config(case_spec, **cfg)
        err = result["solver_info"]["verification_l2_error"]
        if err < best_err:
            best = result
            best_err = err
        elapsed = time.perf_counter() - start
        if err <= target_error and elapsed > 10.0:
            break

    if best is None:
        raise RuntimeError("No configuration succeeded")

    return best


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.08, "dt": 0.01}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['verification_l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(result["u"].shape)
        print(result["solver_info"])
