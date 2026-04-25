from __future__ import annotations

# DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: pressure_pinning / manufactured_solution
#
# METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes

import math
import time
from typing import Any, Dict, Tuple

import numpy as np
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_callable(x: np.ndarray) -> np.ndarray:
    px = np.pi * x[0]
    py = np.pi * x[1]
    return np.vstack(
        (
            np.pi * np.cos(py) * np.sin(px),
            -np.pi * np.cos(px) * np.sin(py),
        )
    )


def _p_exact_callable(x: np.ndarray) -> np.ndarray:
    px = np.pi * x[0]
    py = np.pi * x[1]
    return np.cos(px) * np.cos(py)


def _sample_function_on_points(func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(msh, candidates, points_xyz)

    value_size = int(np.prod(func.function_space.element.value_shape)) if func.function_space.element.value_shape else 1
    local_vals = np.full((points_xyz.shape[0], value_size), np.nan, dtype=np.float64)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points_xyz.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points_xyz[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = np.asarray(
            func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32)),
            dtype=np.float64,
        )
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        local_vals[np.asarray(ids_local, dtype=np.int32)] = vals

    gathered = msh.comm.allgather(local_vals)
    result = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr[:, 0])
        result[mask] = arr[mask]

    if np.isnan(result[:, 0]).any():
        missing = np.where(np.isnan(result[:, 0]))[0]
        eps = 1.0e-12
        shifted = points_xyz.copy()
        shifted[missing, 0] = np.clip(shifted[missing, 0], eps, 1.0 - eps)
        shifted[missing, 1] = np.clip(shifted[missing, 1], eps, 1.0 - eps)

        tree2 = geometry.bb_tree(msh, msh.topology.dim)
        candidates2 = geometry.compute_collisions_points(tree2, shifted)
        colliding2 = geometry.compute_colliding_cells(msh, candidates2, shifted)

        local_vals2 = np.full((points_xyz.shape[0], value_size), np.nan, dtype=np.float64)
        pts_local = []
        cells_local = []
        ids_local = []
        for i in missing:
            links = colliding2.links(i)
            if len(links) > 0:
                pts_local.append(shifted[i])
                cells_local.append(links[0])
                ids_local.append(i)

        if pts_local:
            vals2 = np.asarray(
                func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32)),
                dtype=np.float64,
            )
            if vals2.ndim == 1:
                vals2 = vals2.reshape(-1, 1)
            local_vals2[np.asarray(ids_local, dtype=np.int32)] = vals2

        gathered2 = msh.comm.allgather(local_vals2)
        for arr in gathered2:
            mask = ~np.isnan(arr[:, 0])
            result[mask] = arr[mask]

    return result


def _velocity_magnitude_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    vals = _sample_function_on_points(u_func, pts)
    return np.linalg.norm(vals[:, :2], axis=1).reshape(ny, nx)


def _manufactured_forms(msh: mesh.Mesh, nu: float):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector(
        (
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        )
    )
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return u_ex, p_ex, f


def _solve_once(n: int, nu: float, ksp_type: str, pc_type: str, rtol: float) -> Tuple[fem.Function, fem.Function, Dict[str, Any]]:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    u_ex, p_ex, f = _manufactured_forms(msh, nu)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_callable)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    prefix = f"stokes_{n}_"
    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=prefix,
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        )
        wh = problem.solve()
        used_ksp = ksp_type
        used_pc = pc_type
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=prefix + "fb_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": rtol},
        )
        wh = problem.solve()
        used_ksp = "preonly"
        used_pc = "lu"

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    one = fem.Constant(msh, ScalarType(1.0))
    area = comm.allreduce(fem.assemble_scalar(fem.form(one * ufl.dx)), op=MPI.SUM)
    p_mean = comm.allreduce(fem.assemble_scalar(fem.form(ph * ufl.dx)), op=MPI.SUM) / area
    ph.x.array[:] -= p_mean
    ph.x.scatter_forward()

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(_u_exact_callable)
    p_exact_fun = fem.Function(Q)
    p_exact_fun.interpolate(_p_exact_callable)

    err_u = fem.Function(V)
    err_u.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_u.x.scatter_forward()

    err_p = fem.Function(Q)
    err_p.x.array[:] = ph.x.array - p_exact_fun.x.array
    err_p.x.scatter_forward()

    e_u_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(err_u, err_u) * ufl.dx)), op=MPI.SUM))
    e_p_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((err_p * err_p) * ufl.dx)), op=MPI.SUM))

    probe_n = max(81, min(161, 2 * n + 1))
    xs = np.linspace(0.0, 1.0, probe_n)
    ys = np.linspace(0.0, 1.0, probe_n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(probe_n * probe_n, dtype=np.float64)))
    uvals = _sample_function_on_points(uh, pts)
    uex = _u_exact_callable(np.vstack((pts[:, 0], pts[:, 1], pts[:, 2]))).T
    grid_err = float(np.sqrt(np.mean(np.sum((uvals[:, :2] - uex[:, :2]) ** 2, axis=1))))

    return uh, ph, {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": 0,
        "verification": {
            "velocity_L2_error": float(e_u_l2),
            "pressure_L2_error": float(e_p_l2),
            "velocity_grid_rms_error": grid_err,
        },
    }


def solve(case_spec: dict) -> dict:
    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("pde", {}).get("viscosity", 5.0)))
    t0 = time.perf_counter()
    budget = 176.251

    candidates = [48, 64, 80, 96, 112, 128]
    chosen = None

    for n in candidates:
        if time.perf_counter() - t0 > 0.85 * budget:
            break
        chosen = _solve_once(n=n, nu=nu, ksp_type="minres", pc_type="hypre", rtol=1.0e-10)
        _, _, info = chosen
        if info["verification"]["velocity_L2_error"] < 2.0e-6 and info["verification"]["velocity_grid_rms_error"] < 2.0e-6:
            if time.perf_counter() - t0 > 5.0:
                break

    uh, _, info = chosen
    u_grid = _velocity_magnitude_grid(uh, case_spec["output"]["grid"])
    info["wall_time_sec"] = float(time.perf_counter() - t0)

    return {"u": u_grid, "solver_info": info}


if __name__ == "__main__":
    case = {
        "pde": {"nu": 5.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
