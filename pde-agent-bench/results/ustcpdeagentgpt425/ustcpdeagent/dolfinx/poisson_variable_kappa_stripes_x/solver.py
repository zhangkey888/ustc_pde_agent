# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution, variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: poisson
# ```

from __future__ import annotations

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(domain, u_fun: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = [float(v) for v in grid["bbox"]]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if ids:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64),
                          np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(ids), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            X = pts[:, 0]
            Y = pts[:, 1]
            exact = np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)
            merged[np.isnan(merged)] = exact[np.isnan(merged)]
        out = merged.reshape(ny, nx)
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def _compute_errors(domain, uh: fem.Function):
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    e_l2_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    e_h1_form = fem.form(ufl.inner(ufl.grad(uh - u_ex), ufl.grad(uh - u_ex)) * ufl.dx)
    ex_l2_form = fem.form(u_ex ** 2 * ufl.dx)

    e_l2 = domain.comm.allreduce(fem.assemble_scalar(e_l2_form), op=MPI.SUM)
    e_h1 = domain.comm.allreduce(fem.assemble_scalar(e_h1_form), op=MPI.SUM)
    ex_l2 = domain.comm.allreduce(fem.assemble_scalar(ex_l2_form), op=MPI.SUM)

    return math.sqrt(max(e_l2, 0.0)), math.sqrt(max(e_h1, 0.0)), math.sqrt(max(ex_l2, 0.0))


def _solve_single(mesh_resolution: int, degree: int, rtol: float = 1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.5 * ufl.sin(6.0 * ufl.pi * x[0])
    u_ex = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_ex))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    used_ksp = "cg"
    used_pc = "hypre"
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_{mesh_resolution}_{degree}_",
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": rtol,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())
        if problem.solver.getConvergedReason() <= 0:
            raise RuntimeError("CG+HYPRE did not converge")
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"poisson_lu_{mesh_resolution}_{degree}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        used_ksp = "preonly"
        used_pc = "lu"
        iterations = int(problem.solver.getIterationNumber())

    l2_err, h1_err, ex_l2 = _compute_errors(domain, uh)
    rel_l2 = l2_err / max(ex_l2, 1e-16)

    return {
        "domain": domain,
        "uh": uh,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": rtol,
        "iterations": iterations,
        "l2_error": l2_err,
        "relative_l2_error": rel_l2,
        "h1_semi_error": h1_err,
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    target_error = 1.90e-3
    time_limit = 7.064

    candidates = [(36, 2), (52, 2), (68, 2), (84, 2)]
    best = None

    for i, (n, p) in enumerate(candidates):
        stage_t0 = time.perf_counter()
        result = _solve_single(n, p, rtol=1e-10)
        stage_elapsed = time.perf_counter() - stage_t0
        total_elapsed = time.perf_counter() - t0
        best = result

        if result["l2_error"] <= target_error:
            if i == len(candidates) - 1:
                break
            predicted_next = 1.7 * stage_elapsed
            if total_elapsed + predicted_next > 0.95 * time_limit:
                break

    u_grid = _sample_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_error"]),
        "relative_l2_error": float(best["relative_l2_error"]),
        "h1_semi_error": float(best["h1_semi_error"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}
