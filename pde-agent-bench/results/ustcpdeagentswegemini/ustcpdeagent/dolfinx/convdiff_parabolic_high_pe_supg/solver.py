import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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


def _u_exact(x, t):
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
    p_eval = []
    c_eval = []
    ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_eval.append(points[i])
            c_eval.append(links[0])
            ids.append(i)

    if p_eval:
        vals = u_fun.eval(np.asarray(p_eval, dtype=np.float64), np.asarray(c_eval, dtype=np.int32))
        local_values[np.asarray(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    global_values = np.empty_like(local_values)
    domain.comm.Allreduce(local_values, global_values, op=MPI.MAX)
    global_values[global_values < -1.0e250] = 0.0
    return global_values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    case_spec = _normalize_case(case_spec)
    comm = MPI.COMM_WORLD

    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.08))
    dt_suggested = float(case_spec["pde"]["time"].get("dt", 0.01))

    epsilon = 0.01
    beta_np = np.array([10.0, 4.0], dtype=np.float64)
    wall_limit = 11.706
    target_error = 3.85e-03

    configs = [
        {"mesh_resolution": 48, "degree": 1, "dt": min(dt_suggested, 0.008), "rtol": 1e-8},
        {"mesh_resolution": 64, "degree": 1, "dt": min(dt_suggested, 0.005), "rtol": 1e-9},
        {"mesh_resolution": 80, "degree": 1, "dt": min(dt_suggested, 0.004), "rtol": 1e-9},
    ]

    best = None
    start_total = time.perf_counter()

    for cfg in configs:
        if best is not None and (time.perf_counter() - start_total) > 0.92 * wall_limit:
            break

        N = int(cfg["mesh_resolution"])
        degree = int(cfg["degree"])
        dt = float(cfg["dt"])
        n_steps = max(1, int(round((t_end - t0) / dt)))
        dt = (t_end - t0) / n_steps

        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
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

        init_expr = fem.Expression(_u_exact(x, t_c), V.element.interpolation_points)
        u_n.interpolate(init_expr)

        fdim = domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        u_bc.interpolate(init_expr)
        bc = fem.dirichletbc(u_bc, dofs)

        uex = _u_exact(x, t_c)
        forcing = -uex - eps_c * ufl.div(ufl.grad(uex)) + ufl.dot(beta_c, ufl.grad(uex))

        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-14)
        pe = beta_norm * h / (2.0 * eps_c + 1.0e-14)
        coth_pe = (ufl.exp(2.0 * pe) + 1.0) / (ufl.exp(2.0 * pe) - 1.0 + 1.0e-14)
        tau = h / (2.0 * beta_norm) * (coth_pe - 1.0 / (pe + 1.0e-14))

        strong_res = (u - u_n) / dt_c - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u)) - forcing
        streamline_test = ufl.dot(beta_c, ufl.grad(v))

        a = (
            (u / dt_c) * v * ufl.dx
            + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
            + tau * strong_res * streamline_test * ufl.dx
        )
        L = (
            (u_n / dt_c) * v * ufl.dx
            + forcing * v * ufl.dx
            + tau * forcing * streamline_test * ufl.dx
        )

        a_form = fem.form(a)
        L_form = fem.form(L)

        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        b = petsc.create_vector(L_form.function_spaces)

        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("gmres")
        ksp.getPC().setType("ilu")
        ksp.setTolerances(rtol=float(cfg["rtol"]), atol=1e-12, max_it=2000)

        total_iterations = 0
        failed = False

        for step in range(1, n_steps + 1):
            t_c.value = ScalarType(t0 + step * dt)
            u_bc.interpolate(fem.Expression(_u_exact(x, t_c), V.element.interpolation_points))

            with b.localForm() as b_local:
                b_local.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])

            try:
                ksp.solve(b, u_h.x.petsc_vec)
                u_h.x.scatter_forward()
            except Exception:
                failed = True
                break

            if ksp.getConvergedReason() <= 0:
                failed = True
                break

            total_iterations += int(ksp.getIterationNumber())
            u_n.x.array[:] = u_h.x.array

        if failed:
            continue

        t_c.value = ScalarType(t_end)
        u_exact_fun = fem.Function(V)
        u_exact_fun.interpolate(fem.Expression(_u_exact(x, t_c), V.element.interpolation_points))
        err_fun = fem.Function(V)
        err_fun.x.array[:] = u_h.x.array - u_exact_fun.x.array
        l2_sq_local = fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx))
        l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

        t_c.value = ScalarType(t0)
        u_init_fun = fem.Function(V)
        u_init_fun.interpolate(fem.Expression(_u_exact(x, t_c), V.element.interpolation_points))

        candidate = {
            "u": _sample_to_grid(u_h, case_spec["output"]["grid"]),
            "u_initial": _sample_to_grid(u_init_fun, case_spec["output"]["grid"]),
            "solver_info": {
                "mesh_resolution": N,
                "element_degree": degree,
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "rtol": float(cfg["rtol"]),
                "iterations": total_iterations,
                "dt": dt,
                "n_steps": n_steps,
                "time_scheme": "backward_euler",
                "verification_l2_error": l2_error,
            },
        }

        best = candidate

        elapsed = time.perf_counter() - start_total
        if l2_error <= target_error and elapsed >= 0.6 * wall_limit:
            break

    if best is None:
        raise RuntimeError("All solver configurations failed.")

    return best


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.08, "dt": 0.01}},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case)
    print(result["u"].shape)
    print(result["solver_info"])
