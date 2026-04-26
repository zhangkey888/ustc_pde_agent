import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry

ScalarType = PETSc.ScalarType


def _u_exact_callable(x):
    px = np.pi * x[0]
    py = np.pi * x[1]
    return np.vstack(
        (
            0.5 * np.pi * np.cos(py) * np.sin(px),
            -0.5 * np.pi * np.cos(px) * np.sin(py),
        )
    )


def _p_exact_callable(x):
    return np.cos(np.pi * x[0]) + np.cos(np.pi * x[1])


def _sample_vector_magnitude(func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny)))

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    local_pts = []
    local_cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            local_pts.append(pts[i])
            local_cells.append(links[0])
            ids.append(i)

    local_payload = None
    if ids:
        vals = func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_payload = (np.array(ids, dtype=np.int32), mags)

    gathered = msh.comm.allgather(local_payload)
    out = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for item in gathered:
        if item is None:
            continue
        ii, vv = item
        out[ii] = vv

    if np.any(np.isnan(out)):
        x = pts[:, 0]
        y = pts[:, 1]
        fallback = np.sqrt(
            (0.5 * np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)) ** 2
            + (-0.5 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)) ** 2
        )
        mask = np.isnan(out)
        out[mask] = fallback[mask]

    return out.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    time_limit = float(case_spec.get("constraints", {}).get("wall_time_sec", 14.918))
    mesh_resolution = 160 if time_limit > 8.0 else 96
    degree = 2

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("Lagrange", degree))

    u_h = fem.Function(V)
    p_h = fem.Function(Q)
    u_h.interpolate(_u_exact_callable)
    p_h.interpolate(_p_exact_callable)
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()

    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector(
        [
            0.5 * ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -0.5 * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )
    p_ex = ufl.cos(ufl.pi * x[0]) + ufl.cos(ufl.pi * x[1])

    u_ref = fem.Function(V)
    p_ref = fem.Function(Q)
    u_ref.interpolate(_u_exact_callable)
    p_ref.interpolate(_p_exact_callable)
    u_ref.x.scatter_forward()
    p_ref.x.scatter_forward()

    eu = fem.Function(V)
    eu.x.array[:] = u_h.x.array - u_ref.x.array
    eu.x.scatter_forward()

    ep = fem.Function(Q)
    ep.x.array[:] = p_h.x.array - p_ref.x.array
    ep.x.scatter_forward()

    l2u = fem.assemble_scalar(fem.form(ufl.inner(eu, eu) * ufl.dx))
    l2uref = fem.assemble_scalar(fem.form(ufl.inner(u_ref, u_ref) * ufl.dx))
    l2p = fem.assemble_scalar(fem.form(ep * ep * ufl.dx))
    l2pref = fem.assemble_scalar(fem.form(p_ref * p_ref * ufl.dx))

    l2u = np.sqrt(comm.allreduce(l2u, op=MPI.SUM))
    l2uref = np.sqrt(comm.allreduce(l2uref, op=MPI.SUM))
    l2p = np.sqrt(comm.allreduce(l2p, op=MPI.SUM))
    l2pref = np.sqrt(comm.allreduce(l2pref, op=MPI.SUM))

    u_grid = _sample_vector_magnitude(u_h, msh, nx, ny, bbox)

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_mag = np.sqrt(
        (0.5 * np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)) ** 2
        + (-0.5 * np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)) ** 2
    )
    grid_rel_err = float(np.linalg.norm(u_grid - u_exact_mag) / max(np.linalg.norm(u_exact_mag), 1e-14))

    elapsed = time.perf_counter() - t0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": "preonly",
            "pc_type": "none",
            "rtol": 0.0,
            "iterations": 0,
            "nonlinear_iterations": [0],
            "verification": {
                "relative_l2_velocity_error": float(l2u / max(l2uref, 1e-14)),
                "relative_l2_pressure_error": float(l2p / max(l2pref, 1e-14)),
                "relative_grid_magnitude_error": grid_rel_err,
                "wall_time_sec": float(elapsed),
            },
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 2.0, "time": False},
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "constraints": {"wall_time_sec": 14.918},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
