import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _get_case_time(case_spec: dict) -> Tuple[float, float, float]:
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.08)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.004)))
    if dt <= 0:
        dt = 0.004
    return t0, t_end, dt


def _choose_discretization(case_spec: dict) -> Tuple[int, int, float]:
    t0, t_end, dt_suggested = _get_case_time(case_spec)
    # Use a finer setup than the suggestion because manufactured solution has high x-frequency (8*pi).
    # Stay well inside the runtime budget with P2 on a moderately fine mesh and smaller dt.
    mesh_resolution = 112
    element_degree = 2
    dt = min(dt_suggested, 0.001)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    return mesh_resolution, element_degree, dt


def _exact_callable(t: float):
    def f(x):
        return np.exp(-t) * np.sin(8.0 * np.pi * x[0]) * np.sin(np.pi * x[1])
    return f


def _sample_function_on_grid(domain, uh: fem.Function, grid_spec: dict) -> np.ndarray:
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

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    map_idx = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            map_idx.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(map_idx, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        # Boundary points can be missed on some partitions; use exact zero fallback only if absolutely needed.
        # For this problem all output points are in the unit square, so this should rarely trigger.
        merged = np.nan_to_num(merged)

    return merged.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, _ = _get_case_time(case_spec)
    mesh_resolution, degree, dt = _choose_discretization(case_spec)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    t_wall_start = time.perf_counter()

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    kappa = fem.Constant(domain, ScalarType(1.0))

    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = (
        (-1.0 + 65.0 * ufl.pi * ufl.pi) * ufl.exp(-t_c)
        * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    )

    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    g_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    u_n = fem.Function(V)
    u_n.interpolate(_exact_callable(t0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_callable(t0))
    u_bc.x.scatter_forward()

    f_fun = fem.Function(V)
    f_fun.interpolate(f_expr)
    f_fun.x.scatter_forward()

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

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
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    total_iterations = 0

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)

        u_bc.interpolate(g_expr)
        u_bc.x.scatter_forward()

        f_fun.interpolate(f_expr)
        f_fun.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its if its >= 0 else 0)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    # Accuracy verification
    u_ex_T = fem.Function(V)
    u_ex_T.interpolate(_exact_callable(t_end))
    u_ex_T.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_T.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_ex_T, u_ex_T) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / l2_ref if l2_ref > 0 else l2_err

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)
    u_initial = _sample_function_on_grid(domain, fem.Function(V), grid_spec)
    # Fill sampled initial condition correctly
    u0_fun = fem.Function(V)
    u0_fun.interpolate(_exact_callable(t0))
    u0_fun.x.scatter_forward()
    u_initial = _sample_function_on_grid(domain, u0_fun, grid_spec)

    wall = time.perf_counter() - t_wall_start

    result = {
        "u": u_grid,
        "u_initial": u_initial,
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
            "relative_l2_error": float(rel_l2_err),
            "wall_time_sec": float(wall),
        },
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.08, "dt": 0.004},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
