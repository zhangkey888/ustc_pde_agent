import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = {
    "equation_type": "heat",
    "spatial_dim": 2,
    "domain_geometry": "rectangle",
    "unknowns": "scalar",
    "coupling": "none",
    "linearity": "linear",
    "time_dependence": "transient",
    "stiffness": "stiff",
    "dominant_physics": "diffusion",
    "peclet_or_reynolds": "N/A",
    "solution_regularity": "smooth",
    "bc_type": "all_dirichlet",
    "special_notes": "manufactured_solution",
}

METHOD = {
    "spatial_method": "fem",
    "element_or_basis": "Lagrange_P1",
    "stabilization": "none",
    "time_method": "backward_euler",
    "nonlinear_solver": "none",
    "linear_solver": "cg",
    "preconditioner": "hypre",
    "special_treatment": "none",
    "pde_skill": "heat",
}


def _extract_time(case_spec: dict):
    time_spec = case_spec.get("time", {})
    pde_spec = case_spec.get("pde", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", pde_spec.get("t_end", 0.1)))
    dt = float(time_spec.get("dt", pde_spec.get("dt", 0.01)))
    scheme = time_spec.get("scheme", "backward_euler")
    return t0, t_end, dt, scheme


def _exact_ufl(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_ufl(msh, kappa, t):
    x = ufl.SpatialCoordinate(msh)
    uex = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    return ufl.diff(uex, t) - ufl.div(kappa * ufl.grad(uex))


def _sample_on_grid(u_func, grid_spec):
    msh = u_func.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Point sampling failed for some output points.")
        return merged.reshape((ny, nx))
    return None


def _run_heat(case_spec: dict, mesh_resolution: int, degree: int, dt_in: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    coeffs = case_spec.get("coefficients", {})
    kappa_value = float(coeffs.get("kappa", 1.0))
    t0, t_end, _, scheme = _extract_time(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    n_steps = max(1, int(round((t_end - t0) / dt_in)))
    dt = (t_end - t0) / n_steps

    kappa = fem.Constant(msh, ScalarType(kappa_value))
    dt_c = fem.Constant(msh, ScalarType(dt))
    t_c = fem.Constant(msh, ScalarType(t0))

    u_n = fem.Function(V)
    u0_expr = fem.Expression(_exact_ufl(msh, t_c), V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array
    u_init.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(u0_expr)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f_expr = _forcing_ufl(msh, kappa, t_c)

    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    iterations_total = 0

    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)
        bc_expr = fem.Expression(_exact_ufl(msh, t_c), V.element.interpolation_points)
        u_bc.interpolate(bc_expr)

        with b.localForm() as b_local:
            b_local.set(0.0)
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
        iterations_total += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(_exact_ufl(msh, t_c), V.element.interpolation_points))

    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_exact.x.array
    err.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(uh, grid_spec)
    u_initial = _sample_on_grid(u_init, grid_spec)

    result = None
    if comm.rank == 0:
        result = {
            "u": u_grid,
            "u_initial": u_initial,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": solver.getType(),
                "pc_type": solver.getPC().getType(),
                "rtol": float(solver.getTolerances()[0]),
                "iterations": int(iterations_total),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": "backward_euler",
                "l2_error": float(l2_error),
            },
        }
    return result


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    time_budget = 2.829

    candidates = [
        (24, 1, 0.01),
        (32, 1, 0.01),
        (40, 1, 0.01),
        (48, 1, 0.005),
        (56, 1, 0.005),
        (48, 2, 0.01),
    ]

    best = None
    best_err = None
    t_start = time.perf_counter()

    for mesh_resolution, degree, dt in candidates:
        iter_start = time.perf_counter()
        current = _run_heat(case_spec, mesh_resolution, degree, dt)
        iter_elapsed = time.perf_counter() - iter_start
        total_elapsed = time.perf_counter() - t_start

        if comm.rank == 0:
            err = current["solver_info"]["l2_error"]
            if best is None or err < best_err:
                best = current
                best_err = err

        total_elapsed = comm.bcast(total_elapsed if comm.rank == 0 else None, root=0)
        iter_elapsed = comm.bcast(iter_elapsed if comm.rank == 0 else None, root=0)

        if time_budget - total_elapsed < max(0.2, 1.2 * iter_elapsed):
            break

    if comm.rank == 0:
        best["solver_info"].pop("l2_error", None)
        return best
    return {"u": None, "u_initial": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "heat", "time": True},
        "coefficients": {"kappa": 1.0},
        "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
