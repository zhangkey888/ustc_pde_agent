import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_on_proc.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals)).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(idx_on_proc, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
    else:
        global_vals = None
    global_vals = comm.bcast(global_vals, root=0)
    return global_vals.reshape(ny, nx)


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    wall_start = time.perf_counter()
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    coeffs = pde.get("coefficients", {})

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.08))
    dt_suggested = float(time_spec.get("dt", 0.004))
    time_scheme = str(time_spec.get("scheme", "backward_euler"))
    if time_scheme != "backward_euler":
        time_scheme = "backward_euler"

    kappa_val = float(coeffs.get("kappa", 5.0))
    time_budget = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 7.561)))

    degree = 1
    if time_budget >= 6.0:
        mesh_resolution = 72
        dt = min(dt_suggested, 0.002)
    else:
        mesh_resolution = 56
        dt = dt_suggested

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa = fem.Constant(domain, ScalarType(kappa_val))

    mode = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact_ufl = ufl.exp(-t_const) * mode
    f_ufl = (5.0 * kappa * ufl.pi**2 - 1.0) * u_exact_ufl

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-t0) * np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))
    u_initial_grid = _sample_function_on_grid(domain, u_n, case_spec["output"]["grid"])

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_const.value = ScalarType(t_now)
        u_bc.interpolate(lambda X, tt=t_now: np.exp(-tt) * np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = uh.x.array

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: np.exp(-t_end) * np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

    err_sq_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    norm_sq_local = fem.assemble_scalar(fem.form(u_ex ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))
    rel_l2 = err_l2 / math.sqrt(comm.allreduce(norm_sq_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])
    elapsed = time.perf_counter() - wall_start

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": float(1e-10),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": time_scheme,
            "l2_error": float(err_l2),
            "relative_l2_error": float(rel_l2),
            "wall_time_sec": float(elapsed),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.08, "dt": 0.004, "scheme": "backward_euler"},
            "coefficients": {"kappa": 5.0},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 7.561,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
