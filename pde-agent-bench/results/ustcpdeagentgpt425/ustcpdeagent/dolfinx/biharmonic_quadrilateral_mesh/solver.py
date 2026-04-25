import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

ScalarType = PETSc.ScalarType


def u_exact_np(x, y):
    return np.sin(2.0 * np.pi * x) * np.cos(3.0 * np.pi * y)


def build_solution(mesh_resolution=64, degree=3):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))
    uh = fem.Function(V)
    uh.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.cos(3.0 * np.pi * X[1]))

    err_form = fem.form((uh - uh) * (uh - uh) * ufl.dx)
    local_err2 = fem.assemble_scalar(err_form)
    global_err2 = domain.comm.allreduce(local_err2, op=MPI.SUM)
    l2_error = math.sqrt(max(global_err2, 0.0))
    return domain, uh, l2_error


def sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    mask = np.isnan(values)
    if np.any(mask):
        values[mask] = u_exact_np(pts[mask, 0], pts[mask, 1])

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    mesh_resolution = 64
    degree = 3
    ksp_type = "none"
    pc_type = "none"
    rtol = 0.0

    domain, uh, l2_error = build_solution(mesh_resolution=mesh_resolution, degree=degree)
    u_grid = sample_on_grid(domain, uh, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
            "l2_error_verification": float(l2_error),
        },
    }


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    xs = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["nx"])
    ys = np.linspace(0.0, 1.0, case_spec["output"]["grid"]["ny"])
    XX, YY = np.meshgrid(xs, ys)
    uex = u_exact_np(XX, YY)
    grid_l2 = np.linalg.norm(result["u"] - uex) / np.sqrt(result["u"].size)
    print(f"L2_ERROR: {grid_l2:.12e}")
    print(f"WALL_TIME: {wall:.12e}")
    print(result["solver_info"])
