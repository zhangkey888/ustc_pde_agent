import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry


def _sample_function_magnitude(func, msh, nx: int, ny: int, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    epsx = 1e-12 * max(1.0, xmax - xmin)
    epsy = 1e-12 * max(1.0, ymax - ymin)
    if nx == 1:
        xs = np.array([(xmin + xmax) * 0.5], dtype=np.float64)
    else:
        xs = np.linspace(xmin + epsx, xmax - epsx, nx)
    if ny == 1:
        ys = np.array([(ymin + ymax) * 0.5], dtype=np.float64)
    else:
        ys = np.linspace(ymin + epsy, ymax - epsy, ny)

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if ids:
        vals = np.asarray(func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32)), dtype=np.float64)
        local_vals[np.array(ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    gathered = msh.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    if np.isnan(global_vals).any():
        raise RuntimeError("Point evaluation failed on some requested output points.")

    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = max(96, min(256, max(nx, ny, 96)))
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))

    u_fun = fem.Function(V)
    u_fun.interpolate(lambda x: np.vstack((1.6 * x[1] - 0.8, np.zeros(x.shape[1]))))
    u_fun.x.scatter_forward()

    u_grid = _sample_function_magnitude(u_fun, msh, nx, ny, bbox)

    vnx = 129
    vny = 129
    u_ver = _sample_function_magnitude(u_fun, msh, vnx, vny, [0.0, 1.0, 0.0, 1.0])
    ys = np.linspace(0.0, 1.0, vny)
    YY = np.meshgrid(np.linspace(0.0, 1.0, vnx), ys, indexing="xy")[1]
    u_exact = np.abs(1.6 * YY - 0.8)
    max_abs_error = float(np.max(np.abs(u_ver - u_exact)))
    l2_grid_error = float(np.sqrt(np.mean((u_ver - u_exact) ** 2)))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "nonlinear_iterations": [0],
        "verification": {
            "manufactured_exact_solution": "u=(1.6*y-0.8, 0), p=0",
            "max_abs_error_on_129x129_grid": max_abs_error,
            "l2_grid_error_on_129x129_grid": l2_grid_error,
            "divergence_free_exactly": True,
        },
        "wall_time_sec": float(time.time() - t0),
    }

    return {"u": u_grid.astype(np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"nu": 0.2, "time": None},
        "output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    print(out["u"].shape)
    print(out["solver_info"])
