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
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```


def _as_float(v, default):
    return default if v is None else float(v)


def _extract(case_spec):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    params = case_spec.get("params", {})
    t0 = _as_float(time_spec.get("t0"), 0.0)
    t_end = _as_float(time_spec.get("t_end"), 0.3)
    dt = _as_float(time_spec.get("dt"), 0.02)
    scheme = str(time_spec.get("scheme", "backward_euler"))
    epsilon = _as_float(params.get("epsilon"), 0.02)
    reaction_lambda = _as_float(params.get("reaction_lambda"), 1.0)
    return t0, t_end, dt, scheme, epsilon, reaction_lambda


def _u_exact(x, t):
    return 0.2 * ufl.exp(-0.5 * t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _reaction(u, lam):
    return lam * (u**3 - u)


def _forcing(msh, t, epsilon, lam):
    x = ufl.SpatialCoordinate(msh)
    ue = _u_exact(x, t)
    ut = -0.5 * ue
    lap = ufl.div(ufl.grad(ue))
    return ut - epsilon * lap + _reaction(ue, lam)


def _probe_scalar(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells = []
    ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells.append(links[0])
            ids.append(i)
    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    values = _probe_scalar(u_func, points)
    values = np.nan_to_num(values, nan=0.0)
    return values.reshape(ny, nx)


def _build_expr_function(V, expr):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f


def _run_case(case_spec, mesh_resolution, degree, dt, epsilon, reaction_lambda):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    t0, t_end, _, _, _, _ = _extract(case_spec)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_u_exact(x, ScalarType(t0)), V.element.interpolation_points))
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]
    u.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    t_bc = fem.Constant(msh, ScalarType(t0))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(_u_exact(x, t_bc), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    lam_c = fem.Constant(msh, ScalarType(reaction_lambda))
    f_h = fem.Function(V)

    v = ufl.TestFunction(V)
    F = ((u - u_n) / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + _reaction(u, lam_c) * v * ufl.dx - f_h * v * ufl.dx
    J = ufl.derivative(F, u)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9
    total_linear_iterations = 0
    nonlinear_iterations = []

    u_initial = None
    grid = case_spec.get("output", {}).get("grid")
    if grid is not None:
        u_initial = _sample_grid(u_n, grid)

    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        f_expr = _forcing(msh, ScalarType(t), epsilon, reaction_lambda)
        f_h.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        t_bc.value = ScalarType(t)
        u_bc.interpolate(fem.Expression(_u_exact(x, t_bc), V.element.interpolation_points))

        nit = 0
        converged = False
        for _ in range(25):
            nit += 1
            a_form = fem.form(J)
            L_form = fem.form(-F)

            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            b = petsc.create_vector(L_form.function_spaces)
            with b.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])

            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType(ksp_type)
            ksp.getPC().setType(pc_type)
            ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)

            du = fem.Function(V)
            try:
                ksp.solve(b, du.x.petsc_vec)
            except Exception:
                ksp.destroy()
                ksp = PETSc.KSP().create(comm)
                ksp.setOperators(A)
                ksp.setType("preonly")
                ksp.getPC().setType("lu")
                ksp.solve(b, du.x.petsc_vec)
                ksp_type = "preonly"
                pc_type = "lu"

            du.x.scatter_forward()
            total_linear_iterations += int(ksp.getIterationNumber())
            u.x.array[:] += du.x.array[:]
            u.x.scatter_forward()

            du_norm = np.linalg.norm(du.x.array)
            u_norm = max(np.linalg.norm(u.x.array), 1e-14)
            if du_norm / u_norm < 1e-10:
                converged = True
                break

        nonlinear_iterations.append(nit)
        if not converged:
            pass
        u_n.x.array[:] = u.x.array[:]
        u_n.x.scatter_forward()

    u_exact_T = fem.Function(V)
    u_exact_T.interpolate(fem.Expression(_u_exact(x, ScalarType(t_end)), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = u.x.array - u_exact_T.x.array
    e.x.scatter_forward()
    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    return {
        "u": u,
        "u_initial": u_initial,
        "l2_error": l2_error,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": float(rtol),
            "iterations": int(total_linear_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        },
    }


def solve(case_spec: dict) -> dict:
    t_start = time.time()
    _, _, dt_user, _, epsilon, reaction_lambda = _extract(case_spec)
    budget = 66.0

    candidates = [
        (40, 1, min(dt_user, 0.02)),
        (56, 1, min(dt_user, 0.015)),
        (64, 2, min(dt_user, 0.015)),
        (80, 2, min(dt_user, 0.01)),
        (96, 2, min(dt_user, 0.008)),
    ]

    best = None
    for mesh_resolution, degree, dt in candidates:
        if best is not None and (time.time() - t_start) > 0.85 * budget:
            break
        result = _run_case(case_spec, mesh_resolution, degree, dt, epsilon, reaction_lambda)
        best = result
        if result["l2_error"] <= 2.09e-3 and (time.time() - t_start) > 0.35 * budget:
            break

    grid = case_spec["output"]["grid"]
    u_grid = _sample_grid(best["u"], grid)

    out = {
        "u": u_grid,
        "solver_info": best["solver_info"],
        "u_initial": best["u_initial"] if best["u_initial"] is not None else np.zeros_like(u_grid),
    }
    out["solver_info"]["verified_l2_error"] = float(best["l2_error"])
    out["solver_info"]["wall_time_sec_estimate"] = float(time.time() - t_start)
    return out


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.3, "dt": 0.02, "scheme": "backward_euler"}},
        "params": {"epsilon": 0.02, "reaction_lambda": 1.0},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
