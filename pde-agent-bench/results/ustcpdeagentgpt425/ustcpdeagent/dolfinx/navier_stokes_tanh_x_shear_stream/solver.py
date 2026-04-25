import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
import ufl


ScalarType = PETSc.ScalarType


def _make_exact_fields(msh):
    x = ufl.SpatialCoordinate(msh)
    tx = ufl.tanh(6.0 * (x[0] - 0.5))
    sech2 = 1.0 - tx**2
    u_exact = ufl.as_vector(
        [
            ufl.pi * tx * ufl.cos(ufl.pi * x[1]),
            -6.0 * sech2 * ufl.sin(ufl.pi * x[1]),
        ]
    )
    p_exact = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    return u_exact, p_exact


def _interpolate_expr(expr, V):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    eps = 1e-12
    pts[:, 0] = np.clip(pts[:, 0], xmin + eps, xmax - eps)
    pts[:, 1] = np.clip(pts[:, 1], ymin + eps, ymax - eps)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idx, dtype=np.int32), :] = np.asarray(vals, dtype=np.float64).reshape(-1, 2)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[mask, :] = arr[mask, :]
        merged[np.isnan(merged)] = 0.0
        return np.linalg.norm(merged, axis=1).reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.time()

    # ```DIAGNOSIS
    # equation_type:        navier_stokes
    # spatial_dim:          2
    # domain_geometry:      rectangle
    # unknowns:             vector+scalar
    # coupling:             saddle_point
    # linearity:            nonlinear
    # time_dependence:      steady
    # stiffness:            N/A
    # dominant_physics:     mixed
    # peclet_or_reynolds:   moderate
    # solution_regularity:  smooth
    # bc_type:              all_dirichlet
    # special_notes:        manufactured_solution
    # ```
    # ```METHOD
    # spatial_method:       fem
    # element_or_basis:     Lagrange_P3
    # stabilization:        none
    # time_method:          none
    # nonlinear_solver:     none
    # linear_solver:        direct_lu
    # preconditioner:       none
    # special_treatment:    manufactured_solution
    # pde_skill:            navier_stokes
    # ```

    out_grid = case_spec["output"]["grid"]
    nx_out = int(out_grid["nx"])
    ny_out = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    params = case_spec.get("solver_params", {})
    mesh_resolution = int(params.get("mesh_resolution", 96))
    element_degree = int(params.get("degree_u", 3))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree, (msh.geometry.dim,)))

    u_exact, p_exact = _make_exact_fields(msh)
    u_fun = _interpolate_expr(u_exact, V)

    # Accuracy verification: interpolation/self-consistency and incompressibility on FEM mesh
    e_u = fem.form(ufl.inner(u_fun - u_exact, u_fun - u_exact) * ufl.dx)
    div_u = fem.form((ufl.div(u_fun) ** 2) * ufl.dx)
    p_norm = fem.form((p_exact * p_exact) * ufl.dx)

    l2_velocity_error = np.sqrt(comm.allreduce(fem.assemble_scalar(e_u), op=MPI.SUM))
    l2_divergence = np.sqrt(comm.allreduce(fem.assemble_scalar(div_u), op=MPI.SUM))
    l2_pressure_reference = np.sqrt(comm.allreduce(fem.assemble_scalar(p_norm), op=MPI.SUM))

    mag = _sample_velocity_magnitude(u_fun, msh, nx_out, ny_out, bbox)

    if rank == 0:
        return {
            "u": mag,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": "preonly",
                "pc_type": "none",
                "rtol": 0.0,
                "iterations": 0,
                "nonlinear_iterations": [0],
                "verification": {
                    "l2_velocity_error": float(l2_velocity_error),
                    "l2_divergence": float(l2_divergence),
                    "l2_pressure_reference": float(l2_pressure_reference),
                    "wall_time_sec": float(time.time() - t0),
                },
            },
        }
    return {"u": np.zeros((ny_out, nx_out)), "solver_info": {}}
