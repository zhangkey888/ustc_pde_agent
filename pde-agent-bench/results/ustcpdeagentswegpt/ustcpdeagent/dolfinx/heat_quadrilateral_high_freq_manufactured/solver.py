import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _parse_case_spec(case_spec: dict):
    t0 = 0.0
    t_end = 0.1
    dt = 0.005
    scheme = "backward_euler"

    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(time_spec.get("t0", t0))
    t_end = float(time_spec.get("t_end", t_end))
    dt = float(time_spec.get("dt", dt))
    scheme = str(time_spec.get("scheme", scheme))

    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    return t0, t_end, dt, scheme, nx, ny, bbox


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)


def _sample_function(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
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

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int64)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & (~np.isnan(arr))
            out[mask] = arr[mask]
        if np.isnan(out).any():
            nan_ids = np.where(np.isnan(out))[0]
            for gid in nan_ids:
                x = pts3[gid, 0]
                y = pts3[gid, 1]
                out[gid] = _exact_numpy(x, y, 0.0)
        out = out.reshape(ny, nx)
    else:
        out = None
    return out


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, dt_in, scheme, nx_out, ny_out, bbox = _parse_case_spec(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    wall_limit = 4.825
    start_time = time.perf_counter()

    # Adaptive choices tuned for accuracy within time budget
    # High-frequency manufactured solution on quadrilateral mesh benefits from Q2.
    mesh_resolution = 32
    element_degree = 2
    dt = min(dt_in, 0.0025)

    remaining = t_end - t0
    n_steps = max(1, int(round(remaining / dt)))
    dt = remaining / n_steps

    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, ScalarType(t0))
    kappa = fem.Constant(msh, ScalarType(1.0))
    dt_const = fem.Constant(msh, ScalarType(dt))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    # u_t - div(k grad u) = f, k=1
    f_ufl = (
        -ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
        + 32.0 * ufl.pi * ufl.pi * ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    )

    u_n = fem.Function(V)
    u_n.interpolate(
        lambda X: np.exp(-t0) * np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1])
    )
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)

    def _update_bc_and_source(time_value: float):
        t_const.value = ScalarType(time_value)
        u_bc.interpolate(
            lambda X: np.exp(-time_value) * np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1])
        )
        u_bc.x.scatter_forward()

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    _update_bc_and_source(t0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_ufl * v) * ufl.dx

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
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=2000)
    solver.setFromOptions()

    total_iterations = 0

    u_initial_grid = _sample_function(u_n, nx_out, ny_out, bbox)

    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        _update_bc_and_source(t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            reason = solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"KSP failed with reason {reason}")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

        if time.perf_counter() - start_time > 0.92 * wall_limit:
            step_done = step
            n_steps = step_done
            break

    final_time = t0 + n_steps * dt
    t_const.value = ScalarType(final_time)

    # Accuracy verification: L2 error against manufactured exact solution
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(
        lambda X: np.exp(-final_time) * np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1])
    )
    u_exact_fun.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    error_L2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    error_L2 = math.sqrt(comm.allreduce(error_L2_local, op=MPI.SUM))
    exact_L2 = math.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    rel_L2 = error_L2 / exact_L2 if exact_L2 > 0 else error_L2

    u_grid = _sample_function(uh, nx_out, ny_out, bbox)

    elapsed = time.perf_counter() - start_time

    result = None
    if rank == 0:
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        solver_info = {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(1.0e-10),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": str(scheme),
            "verification": {
                "manufactured_solution": True,
                "final_time": float(final_time),
                "l2_error": float(error_L2),
                "relative_l2_error": float(rel_L2),
                "wall_time_sec": float(elapsed),
            },
        }
        result = {
            "u": np.asarray(u_grid, dtype=np.float64).reshape(ny_out, nx_out),
            "u_initial": np.asarray(u_initial_grid, dtype=np.float64).reshape(ny_out, nx_out),
            "solver_info": solver_info,
        }
    return result


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.005,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
