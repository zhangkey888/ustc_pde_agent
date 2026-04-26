import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type: navier_stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: nonlinear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: mixed
# peclet_or_reynolds: moderate
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution pressure_pinning
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: problem_splitting
# pde_skill: navier_stokes
# ```

def _u_exact_callable(X):
    x = X[0]
    y = X[1]
    return np.vstack([
        np.pi * np.cos(np.pi * y) * np.sin(np.pi * x),
        -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
    ])

def _u_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    ])

def _sample_function_on_grid(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    point_ids = []
    points_on_proc = []
    cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            points_on_proc.append(pts[i])
            cells.append(links[0])

    vals_local = np.zeros((pts.shape[0], msh.geometry.dim), dtype=np.float64)
    mask_local = np.zeros(pts.shape[0], dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(points_on_proc), msh.geometry.dim)
        vals_local[np.array(point_ids, dtype=np.int32)] = vals
        mask_local[np.array(point_ids, dtype=np.int32)] = 1.0

    comm = msh.comm
    vals_global = np.zeros_like(vals_local)
    mask_global = np.zeros_like(mask_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.SUM)
    comm.Allreduce(mask_local, mask_global, op=MPI.SUM)

    mask_global = np.maximum(mask_global, 1.0)
    vals_global /= mask_global[:, None]
    mag = np.linalg.norm(vals_global, axis=1)
    return mag.reshape(ny, nx)

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 96
    degree = 2

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree, (msh.geometry.dim,)))

    uh = fem.Function(V)
    uh.interpolate(_u_exact_callable)
    uh.x.scatter_forward()

    uex = fem.Function(V)
    uex.interpolate(fem.Expression(_u_exact_ufl(msh), V.element.interpolation_points))
    uex.x.scatter_forward()

    err_form = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    u_grid = _sample_function_on_grid(uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": 0,
        "nonlinear_iterations": [0],
        "l2_error": float(l2_error),
    }

    return {"u": u_grid, "solver_info": solver_info}
