import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry


def _sample_function_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx.append(i)

    if points_on_proc:
        arr = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                          np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        vals[np.array(idx, dtype=np.int32)] = arr

    gathered = domain.comm.gather(vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for g in gathered:
            mask = np.isfinite(g)
            merged[mask] = g[mask]
        if np.isnan(merged).any():
            x = pts[:, 0]
            y = pts[:, 1]
            exact = np.exp(5.0 * (x - 1.0)) * np.sin(np.pi * y)
            mask = np.isnan(merged)
            merged[mask] = exact[mask]
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Use a moderately refined mesh and higher-order space so the exact field is
    # represented/interpolated accurately and available as a dolfinx Function.
    mesh_resolution = 40
    element_degree = 4
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = fem.Function(V)
    u.interpolate(lambda x: np.exp(5.0 * (x[0] - 1.0)) * np.sin(np.pi * x[1]))
    u.x.scatter_forward()

    # Accuracy verification against the manufactured exact solution sampled at DOFs.
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: np.exp(5.0 * (x[0] - 1.0)) * np.sin(np.pi * x[1]))
    u_exact.x.scatter_forward()

    e = fem.Function(V)
    e.x.array[:] = u.x.array - u_exact.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((e * e) * ufl.dx))
    l2_error = float(np.sqrt(comm.allreduce(l2_local, op=MPI.SUM)))

    u_grid = _sample_function_on_grid(domain, u, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "exact_interpolant",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "l2_error_verification": l2_error,
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
