from __future__ import annotations

import math
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _sample_on_grid(domain, uh: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(ids, dtype=np.int32)] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict[str, Any]:
    comm = MPI.COMM_WORLD

    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 48))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    ksp_type = str(case_spec.get("solver", {}).get("ksp_type", "cg"))
    pc_type = str(case_spec.get("solver", {}).get("pc_type", "hypre"))
    rtol = float(case_spec.get("solver", {}).get("rtol", 1.0e-10))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    kappa = ScalarType(1.0)
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 10000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    err_global = comm.allreduce(err_local, op=MPI.SUM)
    l2_error = math.sqrt(max(err_global, 0.0))

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    ksp = problem.solver
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": rtol,
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": float(l2_error),
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {
        "u": np.empty((case_spec["output"]["grid"]["ny"], case_spec["output"]["grid"]["nx"]), dtype=np.float64),
        "solver_info": solver_info,
    }
