import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_callable(t):
    def ufun(x):
        return np.exp(-t) * (
            np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
            + 0.2 * np.sin(6.0 * np.pi * x[0]) * np.sin(5.0 * np.pi * x[1])
        )
    return ufun


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals.real

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & (~np.isnan(arr))
            out[mask] = arr[mask]
        if np.isnan(out).any():
            bad = np.where(np.isnan(out))[0][:10]
            raise RuntimeError(f"Failed to evaluate solution at some output points, sample indices={bad}")
        grid = out.reshape(ny, nx)
    else:
        grid = None
    return domain.comm.bcast(grid, root=0)


def _run_once(mesh_n, degree, dt, epsilon, t0, t_end, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    u_exact_ufl = lambda t: ufl.exp(-t) * (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.2 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
    )
    f_ufl = lambda t: (
        -u_exact_ufl(t)
        + epsilon * (
            2.0 * ufl.pi**2 * ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 12.2 * ufl.pi**2 * ufl.exp(-t) * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
        )
        + u_exact_ufl(t)
    )

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u_n = fem.Function(V)
    u_n.interpolate(_exact_callable(t0))
    u_n.x.scatter_forward()

    u_sol = fem.Function(V)
    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array[:]
    u_init.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    t_nm1 = fem.Constant(domain, ScalarType(t0))
    t_n = fem.Constant(domain, ScalarType(t0 + dt))
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    f_mid = 0.5 * (f_ufl(t_nm1) + f_ufl(t_n))
    g_n = u_exact_ufl(t_n)

    a = (
        (1.0 / dt_c) * u * v * ufl.dx
        + 0.5 * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + 0.5 * u * v * ufl.dx
    )
    L = (
        (1.0 / dt_c) * u_n * v * ufl.dx
        - 0.5 * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - 0.5 * u_n * v * ufl.dx
        + f_mid * v * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=2000)
    solver.setFromOptions()

    total_iterations = 0
    start = time.perf_counter()
    tcur = t0

    for step in range(n_steps):
        t_old = tcur
        tcur = t0 + (step + 1) * dt
        t_nm1.value = ScalarType(t_old)
        t_n.value = ScalarType(tcur)
        u_bc.interpolate(_exact_callable(tcur))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        u_sol.x.array[:] = u_n.x.array[:]
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        try:
            its = solver.getIterationNumber()
        except Exception:
            its = 0
        total_iterations += int(its)
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()

    solve_time = time.perf_counter() - start

    u_ex_T = u_exact_ufl(ScalarType(t_end))
    err_expr = u_sol - u_ex_T
    eL2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(err_expr, err_expr) * ufl.dx))
    eL2_sq = comm.allreduce(eL2_sq_local, op=MPI.SUM)
    eL2 = math.sqrt(max(eL2_sq, 0.0))

    return {
        "domain": domain,
        "u": u_sol,
        "u_initial_fem": u_init,
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": total_iterations,
        "l2_error": eL2,
        "wall_time": solve_time,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    output_grid = case_spec["output"]["grid"]

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.4))
    dt_user = float(time_spec.get("dt", 0.005))

    epsilon = float(case_spec.get("epsilon", pde.get("epsilon", 0.02)))

    degree = 2
    mesh_n = int(case_spec.get("mesh_resolution", 64))
    dt = min(dt_user, 0.005)

    try:
        best = _run_once(
            mesh_n=mesh_n,
            degree=degree,
            dt=dt,
            epsilon=epsilon,
            t0=t0,
            t_end=t_end,
            ksp_type="cg",
            pc_type="hypre",
            rtol=1e-10,
        )
    except Exception:
        best = _run_once(
            mesh_n=mesh_n,
            degree=degree,
            dt=dt,
            epsilon=epsilon,
            t0=t0,
            t_end=t_end,
            ksp_type="preonly",
            pc_type="lu",
            rtol=1e-12,
        )

    u_grid = _sample_on_grid(best["domain"], best["u"], output_grid)
    u_initial_grid = _sample_on_grid(best["domain"], best["u_initial_fem"], output_grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "crank_nicolson",
        "verification_l2_error": float(best["l2_error"]),
        "measured_wall_time": float(best["wall_time"]),
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
        "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(output_grid["ny"], output_grid["nx"]),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.4, "dt": 0.005, "scheme": "crank_nicolson"},
            "epsilon": 0.02,
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
