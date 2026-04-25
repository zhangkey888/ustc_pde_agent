from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


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
# special_notes: variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```


def _kappa(domain):
    x = ufl.SpatialCoordinate(domain)
    return 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5) ** 2)


def _rhs(domain):
    x = ufl.SpatialCoordinate(domain)
    return 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])


def _build_problem(domain, degree: int, ksp_type: str, pc_type: str, rtol: float):
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(_kappa(domain) * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(_rhs(domain), v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_case_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        },
    )
    return V, problem


def _solve_once(nx: int, degree: int, ksp_type: str, pc_type: str, rtol: float):
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, nx, cell_type=mesh.CellType.triangle)
    V, problem = _build_problem(domain, degree, ksp_type, pc_type, rtol)
    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0
    its = int(problem.solver.getIterationNumber())
    return domain, V, uh, elapsed, its, nx


def _estimate_h1_difference(u_coarse: fem.Function, u_fine: fem.Function) -> float:
    V_f = u_fine.function_space
    uc = fem.Function(V_f)
    uc.interpolate(u_coarse)
    diff = fem.Function(V_f)
    diff.x.array[:] = u_fine.x.array - uc.x.array
    diff.x.scatter_forward()
    form = fem.form(ufl.inner(_kappa(V_f.mesh) * ufl.grad(diff), ufl.grad(diff)) * ufl.dx)
    local = fem.assemble_scalar(form)
    global_val = V_f.mesh.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(max(global_val, 0.0))


def _sample_on_grid(uh: fem.Function, grid_spec: Dict) -> np.ndarray:
    domain = uh.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.zeros(nx * ny, dtype=np.float64)
    local_points = []
    local_cells = []
    owners = []

    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            owners.append(i)

    if local_points:
        vals = uh.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        values[np.array(owners, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    domain.comm.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t_start = time.perf_counter()

    degree = 2
    rtol = 1e-10
    ksp_type = "cg"
    pc_type = "hypre"
    time_limit = 24.325
    budget = 0.78 * time_limit

    candidates = [80, 112, 144, 176, 208]

    best = None
    previous = None
    total_iterations = 0
    estimator = None

    for n in candidates:
        if best is not None and (time.perf_counter() - t_start) > budget:
            break
        try:
            current = _solve_once(n, degree, ksp_type, pc_type, rtol)
        except Exception:
            ksp_type = "preonly"
            pc_type = "lu"
            current = _solve_once(n, degree, ksp_type, pc_type, rtol)

        domain, V, uh, solve_time, its, nx_used = current
        total_iterations += its

        if previous is not None:
            estimator = _estimate_h1_difference(previous[2], uh)

        best = current
        previous = current

        elapsed = time.perf_counter() - t_start
        if solve_time > 0.0 and elapsed + 1.8 * solve_time > budget:
            break

    if best is None:
        raise RuntimeError("No successful solve completed.")

    domain, V, uh, _, _, nx_used = best
    u_grid = _sample_on_grid(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(nx_used),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
    }
    if estimator is not None:
        solver_info["accuracy_estimate_h1"] = float(estimator)

    return {"u": u_grid, "solver_info": solver_info}
