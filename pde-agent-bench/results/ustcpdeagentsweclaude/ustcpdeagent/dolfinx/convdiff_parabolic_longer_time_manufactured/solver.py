import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS
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
#
# METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion

ScalarType = PETSc.ScalarType


def _parse_case(case_spec):
    pde = case_spec.get("pde", {})
    time = pde.get("time", case_spec.get("time", {}))
    params = pde.get("parameters", case_spec.get("parameters", {}))
    eps = float(params.get("epsilon", params.get("eps", 0.05)))
    beta = np.array(params.get("beta", [2.0, 1.0]), dtype=np.float64)
    t0 = float(time.get("t0", 0.0))
    t_end = float(time.get("t_end", 0.2))
    dt_suggested = float(time.get("dt", 0.02))
    return eps, beta, t0, t_end, dt_suggested


def _choose_discretization():
    return 56, 1, 0.01, "gmres", "ilu", 1.0e-9


def _build_sampling_grid(case_spec):
    out = case_spec["output"]["grid"]
    nx = int(out["nx"])
    ny = int(out["ny"])
    xmin, xmax, ymin, ymax = map(float, out["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return nx, ny, pts


def _sample_function_on_grid(domain, uh, pts, nx, ny):
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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.allgather(local_vals)
    vals = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(vals) & (~np.isnan(arr))
        vals[mask] = arr[mask]
    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    eps, beta_np, t0, t_end, _dt_suggested = _parse_case(case_spec)
    mesh_resolution, degree, dt, ksp_type, pc_type, rtol = _choose_discretization()

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    eps_c = fem.Constant(domain, ScalarType(eps))
    beta_c = fem.Constant(domain, np.array(beta_np, dtype=ScalarType))

    u_exact = ufl.exp(-2.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = -2.0 * u_exact - eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_c, ufl.grad(u_exact))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)

    def bc_fun(X):
        return np.exp(-2.0 * float(t_const.value)) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1])

    u_bc.interpolate(bc_fun)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-2.0 * t0) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = float(np.linalg.norm(beta_np))
    tau_expr = 1.0 / ufl.sqrt((2.0 / dt) ** 2 + (2.0 * beta_norm / h) ** 2 + (9.0 * 4.0 * eps / (h * h)) ** 2)

    strong_res_u = (1.0 / dt) * u + ufl.dot(beta_c, ufl.grad(u)) - eps_c * ufl.div(ufl.grad(u))
    strong_res_rhs = (1.0 / dt) * u_n + f_expr
    streamline_test = ufl.dot(beta_c, ufl.grad(v))

    a = (
        (1.0 / dt) * u * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau_expr * strong_res_u * streamline_test * ufl.dx
    )
    L = (
        strong_res_rhs * v * ufl.dx
        + tau_expr * strong_res_rhs * streamline_test * ufl.dx
    )

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

    uh = fem.Function(V)

    nx, ny, pts = _build_sampling_grid(case_spec)
    u_initial_grid = _sample_function_on_grid(domain, u_n, pts, nx, ny)

    n_steps = int(round((t_end - t0) / dt))
    iterations = 0

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(bc_fun)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array

    u_ex_fn = fem.Function(V)
    u_ex_fn.interpolate(lambda X: np.exp(-2.0 * t_end) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_ex_fn.x.array
    l2_local = fem.assemble_scalar(fem.form(err * err * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, pts, nx, ny)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error_verification": float(l2_error),
        },
    }
