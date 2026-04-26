from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

DIAGNOSIS_CARD = """
DIAGNOSIS
equation_type: heat
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
""".strip()

METHOD_CARD = """
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: heat
""".strip()


def _exact_callable(t: float):
    def f(x):
        return np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    return f


def _source_ufl(msh, kappa, t_const):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    return -u_ex - ufl.div(kappa * ufl.grad(u_ex))


def _eval_function(u_fun: fem.Function, points: np.ndarray) -> np.ndarray:
    msh = u_fun.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    order = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            order.append(i)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_fun.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(order, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Point evaluation failed for some sample points.")
        return out
    return np.empty(0, dtype=np.float64)


def _sample_to_grid(u_fun: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )
    vals = _eval_function(u_fun, pts)
    if u_fun.function_space.mesh.comm.rank == 0:
        return vals.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]

    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.1)))
    dt_suggested = float(pde.get("dt", case_spec.get("dt", 0.01)))
    kappa_value = float(pde.get("kappa", case_spec.get("kappa", 1.0)))
    scheme = str(pde.get("scheme", case_spec.get("scheme", "backward_euler"))).lower()
    if scheme != "backward_euler":
        scheme = "backward_euler"

    mesh_resolution = int(case_spec.get("mesh_resolution", 56))
    element_degree = int(case_spec.get("element_degree", 2))
    dt = min(dt_suggested, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    start = time.perf_counter()

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(msh, ScalarType(kappa_value))
    dt_c = fem.Constant(msh, ScalarType(dt))
    t_c = fem.Constant(msh, ScalarType(t0))

    u_n = fem.Function(V)
    u_n.interpolate(_exact_callable(t0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_callable(t0))
    u_bc.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    f = _source_ufl(msh, kappa, t_c)

    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

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
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)
    solver.setFromOptions()

    iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        u_bc.interpolate(_exact_callable(t))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        iterations += int(solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(_exact_callable(t_end))
    u_exact.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    u0 = fem.Function(V)
    u0.interpolate(_exact_callable(t0))
    u0.x.scatter_forward()

    u_grid = _sample_to_grid(uh, grid)
    u_initial_grid = _sample_to_grid(u0, grid)

    wall_time = time.perf_counter() - start

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": iterations,
        "dt": float(dt),
        "n_steps": n_steps,
        "time_scheme": scheme,
        "l2_error": float(l2_error),
        "wall_time_sec": float(wall_time),
        "diagnosis_card": DIAGNOSIS_CARD,
        "method_card": METHOD_CARD,
    }

    if rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {
        "u": np.empty((int(grid["ny"]), int(grid["nx"])), dtype=np.float64),
        "u_initial": np.empty((int(grid["ny"]), int(grid["nx"])), dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "t0": 0.0,
            "t_end": 0.1,
            "dt": 0.01,
            "scheme": "backward_euler",
            "kappa": 1.0,
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
