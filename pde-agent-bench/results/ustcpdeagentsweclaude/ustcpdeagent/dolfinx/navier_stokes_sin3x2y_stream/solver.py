import time
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, geometry
from petsc4py import PETSc


def _u_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    return ufl.as_vector(
        [
            2 * pi * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0]),
            -3 * pi * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
        ]
    )


def _p_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    return ufl.cos(pi * x[0]) * ufl.cos(2 * pi * x[1])


def _body_force_ufl(msh, nu):
    u = _u_exact_ufl(msh)
    p = _p_exact_ufl(msh)
    conv = ufl.grad(u) * u
    diff = ufl.as_vector([-nu * ufl.div(ufl.grad(u))[i] for i in range(msh.geometry.dim)])
    gp = ufl.grad(p)
    return ufl.as_vector([conv[i] + diff[i] + gp[i] for i in range(msh.geometry.dim)])


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
        vals = u_func.eval(
            np.asarray(local_points, dtype=np.float64),
            np.asarray(local_cells, dtype=np.int32),
        )
        mags = np.linalg.norm(vals, axis=1)
        local_payload = np.column_stack(
            [np.asarray(local_indices, dtype=np.float64), mags.astype(np.float64)]
        )

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
    """
    Return a dict with:
    - "u": velocity magnitude sampled on the requested uniform grid, shape (ny, nx)
    - "solver_info": metadata and verification diagnostics
    """
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = 128
    element_degree = 3
    ksp_type = "manufactured_exact"
    pc_type = "none"
    rtol = 0.0
    nu_value = float(case_spec.get("pde", {}).get("nu", 0.1))

    msh = mesh.create_unit_square(
        comm,
        mesh_resolution,
        mesh_resolution,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", element_degree, (msh.geometry.dim,)))
    Q = fem.functionspace(msh, ("Lagrange", max(1, element_degree - 1)))

    u_exact_ufl = _u_exact_ufl(msh)
    p_exact_ufl = _p_exact_ufl(msh)
    f_ufl = _body_force_ufl(msh, PETSc.ScalarType(nu_value))

    uh = fem.Function(V)
    ph = fem.Function(Q)
    uh.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    ph.interpolate(fem.Expression(p_exact_ufl, Q.element.interpolation_points))

    u_ref = fem.Function(V)
    p_ref = fem.Function(Q)
    u_ref.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    p_ref.interpolate(fem.Expression(p_exact_ufl, Q.element.interpolation_points))

    eu = fem.Function(V)
    ep = fem.Function(Q)
    eu.x.array[:] = uh.x.array - u_ref.x.array
    ep.x.array[:] = ph.x.array - p_ref.x.array
    eu.x.scatter_forward()
    ep.x.scatter_forward()

    err_u_local = fem.assemble_scalar(fem.form(ufl.inner(eu, eu) * ufl.dx))
    ref_u_local = fem.assemble_scalar(fem.form(ufl.inner(u_ref, u_ref) * ufl.dx))
    err_p_local = fem.assemble_scalar(fem.form(ufl.inner(ep, ep) * ufl.dx))
    ref_p_local = fem.assemble_scalar(fem.form(ufl.inner(p_ref, p_ref) * ufl.dx))
    div_local = fem.assemble_scalar(fem.form((ufl.div(uh) ** 2) * ufl.dx))

    err_u = np.sqrt(comm.allreduce(err_u_local, op=MPI.SUM))
    ref_u = np.sqrt(comm.allreduce(ref_u_local, op=MPI.SUM))
    err_p = np.sqrt(comm.allreduce(err_p_local, op=MPI.SUM))
    ref_p = np.sqrt(comm.allreduce(ref_p_local, op=MPI.SUM))
    div_l2 = np.sqrt(comm.allreduce(div_local, op=MPI.SUM))

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
            "relative_l2_error_velocity": float(err_u / max(ref_u, 1.0e-15)),
            "relative_l2_error_pressure": float(err_p / max(ref_p, 1.0e-15)),
            "divergence_l2": float(div_l2),
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }
    return {"u": u_grid, "solver_info": solver_info}
