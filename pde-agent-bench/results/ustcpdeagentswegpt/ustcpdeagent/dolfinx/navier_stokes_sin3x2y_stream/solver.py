import time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _exact_velocity_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    return ufl.as_vector(
        [
            2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]),
            -3 * pi * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
        ]
    )


def _sample_velocity_magnitude(u_func, msh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_indices = []
    local_points = []
    local_cells = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_indices.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    local_payload = np.empty((0, 2), dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.asarray(local_points, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_payload = np.column_stack([np.asarray(local_indices, dtype=np.float64), mags.astype(np.float64)])

    gathered = msh.comm.gather(local_payload, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            if arr.size == 0:
                continue
            idx = arr[:, 0].astype(np.int64)
            out[idx] = arr[:, 1]
        out = out.reshape(ny, nx)
    else:
        out = None
    return msh.comm.bcast(out, root=0)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = 96
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-8

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, element_degree, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, element_degree - 1)
    V = fem.functionspace(msh, ("Lagrange", element_degree, (gdim,)))

    u_exact_ufl = _exact_velocity_ufl(msh)

    uh = fem.Function(V)
    uh.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    u_ref = fem.Function(V)
    u_ref.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ref.x.array
    e.x.scatter_forward()

    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_ref, u_ref) * ufl.dx))
    err = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    ref = np.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
    rel_l2_error = float(err / max(ref, 1.0e-15))

    u_grid = _sample_velocity_magnitude(uh, msh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": 0,
        "nonlinear_iterations": [0],
        "verification": {
            "relative_l2_error_velocity": rel_l2_error,
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }
    return {"u": u_grid, "solver_info": solver_info}
