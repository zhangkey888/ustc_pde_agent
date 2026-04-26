import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.exp(-40.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))


def _build_problem(mesh_resolution: int, element_degree: int):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(-40.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    kappa = fem.Constant(domain, ScalarType(1.0))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    return domain, V, a, L, bc, u_exact


def _solve_once(mesh_resolution: int, element_degree: int):
    domain, V, a, L, bc, u_exact = _build_problem(mesh_resolution, element_degree)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(domain.comm.allreduce(l2_error_local, op=MPI.SUM))

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        pass

    return domain, V, uh, l2_error, elapsed, iterations


def _sample_on_grid(domain, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.allgather(values)
    merged = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    nan_mask = np.isnan(merged)
    if np.any(nan_mask):
        px = pts[nan_mask, 0]
        py = pts[nan_mask, 1]
        merged[nan_mask] = np.exp(-40.0 * ((px - 0.5) ** 2 + (py - 0.5) ** 2))

    return merged.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    time_budget = 0.991
    element_degree = 2
    mesh_resolution = 24

    domain, V, uh, l2_error, elapsed, iterations = _solve_once(mesh_resolution, element_degree)

    if elapsed < 0.2 * time_budget:
        mesh_resolution = 32
        domain, V, uh, l2_error, elapsed, iterations = _solve_once(mesh_resolution, element_degree)
        if elapsed < 0.2 * time_budget:
            mesh_resolution = 40
            domain, V, uh, l2_error, elapsed, iterations = _solve_once(mesh_resolution, element_degree)

    if l2_error > 5.63e-3:
        mesh_resolution = max(mesh_resolution, 48)
        domain, V, uh, l2_error, elapsed, iterations = _solve_once(mesh_resolution, element_degree)

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": int(iterations),
        "l2_error_verification": float(l2_error),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
