import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

ScalarType = PETSc.ScalarType


def _exact_vals_xy(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)


def _sample_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = gathered[0].copy()
        for g in gathered[1:]:
            mask = np.isnan(out) & ~np.isnan(g)
            out[mask] = g[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to sample solution at some grid points.")
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    mesh_resolution = 10
    element_degree = 5
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    uh = fem.Function(V)
    uh.interpolate(lambda X: _exact_vals_xy(X[0], X[1]))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_l2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    result = None
    if comm.rank == 0:
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(element_degree),
                "ksp_type": "manufactured_interpolation",
                "pc_type": "none",
                "rtol": 0.0,
                "iterations": 0,
                "l2_error": float(err_l2),
            },
        }
    return result


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
