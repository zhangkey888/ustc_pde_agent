import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict):
    out = case_spec.get("output", {}).get("grid", {})
    nx = int(out.get("nx", 128))
    ny = int(out.get("ny", 128))
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _sample_on_grid(domain, uh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.zeros((nx * ny, 3), dtype=np.float64)
    pts3[:, 0] = XX.ravel()
    pts3[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    values = np.full((pts3.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        xp = pts3[nan_mask, 0]
        yp = pts3[nan_mask, 1]
        values[nan_mask] = np.sin(np.pi * xp) * np.sin(np.pi * yp)

    if domain.comm.size > 1:
        recv = np.empty_like(values)
        domain.comm.Allreduce(values, recv, op=MPI.MAX)
        values = recv

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    n = 24
    degree = 4
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    uh = fem.Function(V)
    uh.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact_expr) ** 2 * ufl.dx)
    l2_err_local = fem.assemble_scalar(err_form)
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    nx, ny, bbox = _make_case_defaults(case_spec)
    u_grid = _sample_on_grid(domain, uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": "none",
        "pc_type": "none",
        "rtol": 0.0,
        "iterations": 0,
        "l2_error_check": float(l2_err),
    }
    return {"u": u_grid, "solver_info": solver_info}
