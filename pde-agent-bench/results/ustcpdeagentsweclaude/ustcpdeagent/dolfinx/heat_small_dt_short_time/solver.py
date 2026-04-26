import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_idx = []
    local_points = []
    local_cells = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_points.append(points[i])
            local_cells.append(links[0])

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        values[np.array(local_idx, dtype=np.int32)] = np.real(vals).reshape(-1)

    gathered = msh.comm.allgather(values)
    out = np.full(points.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(out) & (~np.isnan(arr))
        out[mask] = arr[mask]
    return out


def _solve_once(mesh_resolution, degree, dt, t0, t_end, kappa=1.0):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, ScalarType(t0))
    kappa_c = fem.Constant(msh, ScalarType(kappa))
    dt_c = fem.Constant(msh, ScalarType(dt))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f_ufl = (-1.0 + 32.0 * ufl.pi * ufl.pi) * u_exact_ufl

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_bc.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    uh.name = "u"

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    u_initial = u_n.x.array.copy()

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except RuntimeError:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setUp()
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its if its >= 0 else 0)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    t_const.value = ScalarType(t_end)
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_ex.x.scatter_forward()

    err_sq = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    norm_sq = fem.assemble_scalar(fem.form((u_ex) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_sq, op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(norm_sq, op=MPI.SUM))
    rel_l2 = err_l2 / max(norm_l2, 1e-16)

    return {
        "mesh": msh,
        "V": V,
        "u": uh,
        "u_initial_coeffs": u_initial,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "iterations": total_iterations,
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1e-10,
        "err_l2": err_l2,
        "rel_l2": rel_l2,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.06))
    dt_suggested = float(pde.get("dt", 0.003))

    total_time_window = max(t_end - t0, 1e-14)
    n_steps = max(1, int(round(total_time_window / dt_suggested)))
    dt = total_time_window / n_steps

    start = time.perf_counter()
    best = _solve_once(80, 2, dt, t0, t_end, kappa=1.0)
    _elapsed = time.perf_counter() - start

    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, out_grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    u_vals = _probe_function(best["u"], pts).reshape(ny, nx)

    u0_fun = fem.Function(best["V"])
    u0_fun.x.array[:] = best["u_initial_coeffs"]
    u0_fun.x.scatter_forward()
    u0_vals = _probe_function(u0_fun, pts).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": "backward_euler",
        "l2_error": float(best["err_l2"]),
        "relative_l2_error": float(best["rel_l2"]),
    }

    return {
        "u": np.asarray(u_vals, dtype=np.float64),
        "u_initial": np.asarray(u0_vals, dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.06, "dt": 0.003},
        "constraints": {"wall_time_sec": 13.169},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    x = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["nx"])
    y = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["ny"])
    X, Y = np.meshgrid(x, y, indexing="xy")
    u_exact = np.exp(-case_spec["pde"]["t_end"]) * np.sin(4.0 * np.pi * X) * np.sin(4.0 * np.pi * Y)
    test_err = float(np.sqrt(np.mean((out["u"] - u_exact) ** 2)))
    print(f"L2_ERROR: {test_err}")
    print(f"WALL_TIME: {wall}")
