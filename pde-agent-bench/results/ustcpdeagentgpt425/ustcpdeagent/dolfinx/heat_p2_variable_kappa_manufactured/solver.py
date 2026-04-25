import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_points_scalar(u_func: fem.Function, points_array: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_points_scalar(u_func, pts)
    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.06)))
    dt_suggested = float(time_spec.get("dt", case_spec.get("dt", 0.01)))

    dt = min(dt_suggested, 0.005)
    n_steps = int(round((t_end - t0) / dt))
    if n_steps <= 0:
        n_steps = 1
    if abs(t0 + n_steps * dt - t_end) > 1e-14:
        n_steps = int(math.ceil((t_end - t0) / dt))
        dt = (t_end - t0) / n_steps

    mesh_resolution = int(case_spec.get("mesh_resolution", 40))
    element_degree = int(case_spec.get("element_degree", 2))
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    pi = math.pi
    u_exact = ufl.exp(-t_const) * ufl.sin(2.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    kappa = 1.0 + 0.4 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    f_expr = ufl.diff(u_exact, t_const) - ufl.div(kappa * ufl.grad(u_exact))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array
    u_h.x.scatter_forward()

    grid_spec = case_spec["output"]["grid"]
    u_initial_grid = _sample_on_uniform_grid(u_n, grid_spec)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_const * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_expr * v) * ufl.dx

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
    solver.setFromOptions()

    total_iterations = 0
    current_time = t0
    wall_t0 = time.perf_counter()

    for _ in range(n_steps):
        current_time += dt
        t_const.value = ScalarType(current_time)
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    wall_elapsed = time.perf_counter() - wall_t0

    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    err_fn = fem.Function(V)
    err_fn.x.array[:] = u_h.x.array - u_exact_final.x.array
    err_fn.x.scatter_forward()

    l2_err_sq = fem.assemble_scalar(fem.form((err_fn * err_fn) * ufl.dx))
    l2_ex_sq = fem.assemble_scalar(fem.form((u_exact_final * u_exact_final) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_err_sq, op=MPI.SUM))
    l2_exact_norm = math.sqrt(comm.allreduce(l2_ex_sq, op=MPI.SUM))
    rel_l2_error = l2_error / (l2_exact_norm + 1e-16)

    u_grid = _sample_on_uniform_grid(u_h, grid_spec)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
            "rel_l2_error": float(rel_l2_error),
            "wall_time_sec": float(wall_elapsed),
        },
    }
