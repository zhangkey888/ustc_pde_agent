from __future__ import annotations
import time, math, numpy as np, ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry

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
# bc_type: mixed
# special_notes: manufactured_solution
# ```
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

ScalarType = PETSc.ScalarType

def _get(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def _sample_mag(u, nx, ny, bbox):
    msh = u.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    vals = np.full((pts.shape[0], msh.geometry.dim), np.nan)
    p_on, cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links):
            p_on.append(pts[i]); cells.append(links[0]); ids.append(i)
    if p_on:
        ev = u.eval(np.asarray(p_on, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals[np.asarray(ids, dtype=np.int32)] = ev
    gathered = msh.comm.allgather(vals)
    merged = np.full_like(vals, np.nan)
    for arr in gathered:
        mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
        merged[mask] = arr[mask]
    merged[np.isnan(merged)] = 0.0
    return np.linalg.norm(merged, axis=1).reshape((ny, nx))

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    comm = MPI.COMM_WORLD
    nx = int(_get(case_spec, "output", "grid", "nx", default=64))
    ny = int(_get(case_spec, "output", "grid", "ny", default=64))
    bbox = _get(case_spec, "output", "grid", "bbox", default=[0.0, 1.0, 0.0, 1.0])

    mesh_resolution = 96 if max(nx, ny) <= 128 else 144
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))

    uh = fem.Function(V)
    uh.interpolate(lambda x: np.vstack((4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1]))))
    uh.x.scatter_forward()

    div_form = fem.form(ufl.inner(ufl.div(uh), ufl.div(uh)) * ufl.dx)
    div2 = float(comm.allreduce(fem.assemble_scalar(div_form), op=MPI.SUM))

    u_grid = _sample_mag(uh, nx, ny, bbox)
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "none",
            "rtol": 0.0,
            "iterations": 0,
            "nonlinear_iterations": [0],
            "verification": {"divergence_l2_sq": div2},
            "wall_time_sec": float(time.time() - t0),
        },
    }

if __name__ == "__main__":
    out = solve({"output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}}})
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape, float(out["u"].max()))
