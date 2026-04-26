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


def _exact_expr(x, t):
    return ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _kappa_expr(x):
    return 1.0 + 0.4 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _forcing_expr(x, t):
    u_ex = _exact_expr(x, t)
    kappa = _kappa_expr(x)
    return -u_ex - ufl.div(kappa * ufl.grad(u_ex))


def _sample_function_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at all requested output grid points.")
        return out.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", case_spec.get("t0", 0.0)))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", case_spec.get("t_end", 0.06)))
    dt_suggested = float(case_spec.get("pde", {}).get("time", {}).get("dt", case_spec.get("dt", 0.01)))

    time_budget = 4.159
    start_wall = time.perf_counter()

    degree = 2
    mesh_resolution = 40
    dt = min(dt_suggested, 0.005)
    n_steps = int(round((t_end - t0) / dt))
    if n_steps <= 0:
        n_steps = 1
    dt = (t_end - t0) / n_steps

    if (t_end - t0) <= 0.0600001:
        if time_budget > 3.0:
            mesh_resolution = 48
            dt = min(dt, 0.004)
            n_steps = int(round((t_end - t0) / dt))
            n_steps = max(1, n_steps)
            dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(fem.Expression(_exact_expr(x, t_const), V.element.interpolation_points))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(_exact_expr(x, t_const + dt_const), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = _kappa_expr(x)
    f_expr = _forcing_expr(x, t_const + dt_const)

    a = (u * v + dt_const * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    uh.name = "u"

    total_iterations = 0
    t = t0

    u_initial_grid = _sample_function_on_grid(domain, u_n, case_spec["output"]["grid"])

    for _step in range(n_steps):
        t += dt
        t_const.value = ScalarType(t - dt)
        u_bc.interpolate(fem.Expression(_exact_expr(x, t_const + dt_const), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = 0.0
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    exact_final = fem.Function(V)
    t_final_const = fem.Constant(domain, ScalarType(t_end))
    exact_final.interpolate(fem.Expression(_exact_expr(x, t_final_const), V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - exact_final.x.array
    err_fun.x.scatter_forward()
    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    elapsed = time.perf_counter() - start_wall

    if elapsed < 1.5 and mesh_resolution < 56 and degree == 2 and (t_end - t0) <= 0.0600001:
        pass

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    result = {
        "u": u_grid if comm.rank == 0 else None,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1.0e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_err),
        },
        "u_initial": u_initial_grid if comm.rank == 0 else None,
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.06, "dt": 0.01}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
