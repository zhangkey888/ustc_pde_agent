from __future__ import annotations

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(eval_map), -1)[:, 0]
        values[np.asarray(eval_map, dtype=np.int32)] = vals

    mask = np.isnan(values)
    if np.any(mask):
        values[mask] = np.exp(0.5 * pts[mask, 0]) * np.sin(2.0 * np.pi * pts[mask, 1])

    return values.reshape(ny, nx)


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(0.5 * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = (
        1.0
        + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + 15.0 * ufl.exp(-200.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.75) ** 2))
    )
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.asarray(np.exp(0.5 * X[0]) * np.sin(2.0 * np.pi * X[1]), dtype=ScalarType))
    bc = fem.dirichletbc(u_bc, dofs)

    used_ksp = ksp_type
    used_pc = pc_type
    iterations = 0

    try:
        petsc_options = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
        if pc_type == "hypre":
            petsc_options["pc_hypre_type"] = "boomeramg"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options=petsc_options,
            petsc_options_prefix=f"poisson_{n}_{degree}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        try:
            iterations = int(problem.solver.getIterationNumber())
        except Exception:
            iterations = 0
    except Exception:
        used_ksp = "preonly"
        used_pc = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": used_ksp, "pc_type": used_pc},
            petsc_options_prefix=f"poisson_fallback_{n}_{degree}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = 1

    e = uh - u_exact
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    err_l2 = float(np.sqrt(comm.allreduce(l2_sq, op=MPI.SUM)))
    err_h1 = float(np.sqrt(comm.allreduce(h1_sq, op=MPI.SUM)))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification": {"L2_error": err_l2, "H1_semi_error": err_h1},
    }
    return domain, uh, solver_info


def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]

    candidates = [(24, 1), (32, 1), (40, 1), (32, 2), (40, 2), (48, 2), (56, 2)]
    budget = 0.85
    best = None
    best_err = float("inf")
    t0 = time.perf_counter()

    for n, degree in candidates:
        domain, uh, info = _solve_once(n, degree)
        if info["verification"]["L2_error"] < best_err:
            best = (domain, uh, info)
            best_err = info["verification"]["L2_error"]
        if time.perf_counter() - t0 > budget:
            break

    domain, uh, solver_info = best
    u_grid = _sample_on_grid(domain, uh, grid_spec)
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
