import math
import time
from typing import Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func: fem.Function, nx: int, ny: int, bbox) -> np.ndarray:
    domain = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values_local[np.array(eval_map, dtype=np.int32)] = vals

    values_global = np.empty_like(values_local)
    domain.comm.Allreduce(values_local, values_global, op=MPI.MAX)
    values_global = np.nan_to_num(values_global, nan=0.0)
    return values_global.reshape(ny, nx)


def _run_case(nx: int, degree: int, dt: float, t_end: float, out_grid: Dict):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps = 0.05
    beta_vec = np.array([2.0, 1.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_vec)
    f_expr = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    u_prev = fem.Function(V)
    u_prev.x.array[:] = 0.0
    u_sol = fem.Function(V)

    h = ufl.CellDiameter(domain)
    beta_norm = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau = h / (2.0 * beta_norm)

    a = (
        u * v * ufl.dx
        + dt * eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + dt * ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * (
            ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v))
            + (1.0 / dt) * u * ufl.dot(beta, ufl.grad(v))
        ) * ufl.dx
    )

    L = (
        u_prev * v * ufl.dx
        + dt * f_expr * v * ufl.dx
        + tau * ((1.0 / dt) * u_prev + f_expr) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=2000)
    solver.setFromOptions()

    n_steps = int(round(t_end / dt))
    total_iterations = 0
    t_start = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        u_prev.x.array[:] = u_sol.x.array

    wall = time.perf_counter() - t_start
    u_grid = _sample_function_on_grid(u_sol, out_grid["nx"], out_grid["ny"], out_grid["bbox"])
    return u_grid, {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-8,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "_wall": float(wall),
    }


def solve(case_spec: dict) -> dict:
    out_grid = case_spec["output"]["grid"]
    pde_time = case_spec.get("pde", {}).get("time", {})
    t_end = float(pde_time.get("t_end", 0.1))
    dt_suggested = float(pde_time.get("dt", 0.02))

    candidates = [
        (64, 1, min(dt_suggested, 0.01)),
        (96, 1, min(dt_suggested, 0.005)),
        (128, 1, min(dt_suggested, 0.0025)),
    ]

    best_u = None
    best_info = None
    previous_u = None
    verification = {}
    budget = 109.011

    for i, (nx, degree, dt) in enumerate(candidates):
        u_grid, info = _run_case(nx, degree, dt, t_end, out_grid)
        if previous_u is not None:
            rel_change = np.linalg.norm(u_grid - previous_u) / max(np.linalg.norm(u_grid), 1e-14)
            verification[f"grid_refinement_rel_change_{i}"] = float(rel_change)
        previous_u = u_grid.copy()
        best_u = u_grid
        best_info = info
        if info["_wall"] > 0.6 * budget:
            break

    best_info.pop("_wall", None)
    best_info["verification"] = verification

    return {
        "u": np.asarray(best_u, dtype=np.float64).reshape(out_grid["ny"], out_grid["nx"]),
        "u_initial": np.zeros((out_grid["ny"], out_grid["nx"]), dtype=np.float64),
        "solver_info": best_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02}},
        "output": {"grid": {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
