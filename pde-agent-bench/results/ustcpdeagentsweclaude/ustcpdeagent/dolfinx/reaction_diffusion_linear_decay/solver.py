# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: preonly
# preconditioner: lu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.cos(2.0 * np.pi * x) * np.sin(np.pi * y)


def _get_params(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.6))
    dt_user = float(time_spec.get("dt", 0.01))
    # Use a more accurate default if not specified; respect user value otherwise.
    dt_target = float(case_spec.get("dt", dt_user if "dt" in time_spec else 0.0025))
    dt_target = min(dt_target, max(t_end - t0, 1e-12))
    n_steps = max(1, int(math.ceil((t_end - t0) / dt_target)))
    dt = (t_end - t0) / n_steps
    mesh_resolution = int(case_spec.get("mesh_resolution", 120))
    degree = int(case_spec.get("element_degree", 2))
    return {
        "t0": t0,
        "t_end": t_end,
        "dt": dt,
        "n_steps": n_steps,
        "scheme": str(time_spec.get("scheme", "backward_euler")),
        "epsilon": float(case_spec.get("epsilon", 0.05)),
        "reaction_alpha": float(case_spec.get("reaction_alpha", 1.0)),
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
    }


def _probe_function(u_func: fem.Function, pts: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        values[np.asarray(idx, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(values)
    out = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    return np.nan_to_num(out, nan=0.0)


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return _probe_function(u_func, pts).reshape((ny, nx))


def solve(case_spec: dict) -> Dict:
    comm = MPI.COMM_WORLD
    params = _get_params(case_spec)
    t0 = params["t0"]
    t_end = params["t_end"]
    dt = params["dt"]
    n_steps = params["n_steps"]
    epsilon = params["epsilon"]
    alpha = params["reaction_alpha"]
    mesh_resolution = params["mesh_resolution"]
    degree = params["element_degree"]

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_np1 = fem.Constant(msh, ScalarType(t0 + dt))
    t_n = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    alpha_c = fem.Constant(msh, ScalarType(alpha))

    u_ex_np1 = ufl.exp(-t_np1) * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_ex_n = ufl.exp(-t_n) * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_ex_np1 = -5.0 * (ufl.pi ** 2) * u_ex_np1
    # Manufactured source consistent with the discrete backward-Euler scheme.
    f_expr = (u_ex_np1 - u_ex_n) / dt_c - eps_c * lap_u_ex_np1 + alpha_c * u_ex_np1

    u_n = fem.Function(V)
    u_ic_expr = ufl.exp(-ScalarType(t0)) * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u_ic_expr, V.element.interpolation_points))
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda z: np.ones(z.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex_np1, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ((1.0 / dt_c) * u * v + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + alpha_c * u * v) * ufl.dx
    L = ((1.0 / dt_c) * u_n * v + f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setTolerances(rtol=1e-12)
    solver.setFromOptions()

    total_iterations = 0
    for step in range(n_steps):
        tn = t0 + step * dt
        tnp1 = tn + dt
        t_n.value = ScalarType(tn)
        t_np1.value = ScalarType(tnp1)
        u_bc.interpolate(fem.Expression(u_ex_np1, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_exact_end = fem.Function(V)
    u_exact_end.interpolate(fem.Expression(u_ex_np1, V.element.interpolation_points))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_exact_end.x.array
    err.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(err * err * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))

    output_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    u_grid = _sample_on_uniform_grid(uh, output_grid)
    u0_grid = _sample_on_uniform_grid(u_initial, output_grid)

    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, output_grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    exact_grid = _exact_numpy(xx, yy, t_end)
    grid_max_error = float(np.max(np.abs(u_grid - exact_grid)))

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
            "grid_max_error": grid_max_error,
            "epsilon": float(epsilon),
            "reaction_alpha": float(alpha),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.6, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
