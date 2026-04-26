import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = vals

    gathered = msh.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
        merged[mask] = arr[mask]

    mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    return mag


def solve(case_spec: dict) -> dict:
    # DIAGNOSIS and METHOD cards required by protocol
    _ = """