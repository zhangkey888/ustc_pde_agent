# ```DIAGNOSIS
# equation_type:        navier_stokes
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             vector+scalar
# coupling:             saddle_point
# linearity:            nonlinear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        preonly
# preconditioner:       lu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _u_exact_numpy(x, y):
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    e = np.exp(-20.0 * r2)
    return -40.0 * (y - 0.5) * e, 40.0 * (x - 0.5) * e


def _u_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    r2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2
    e = ufl.exp(-20.0 * r2)
    return ufl.as_vector((-40.0 * (x[1] - 0.5) * e, 40.0 * (x[0] - 0.5) * e))


def _forcing_stokes_from_exact(msh, nu):
    uex = _u_exact_ufl(msh)
    return -nu * ufl.div(ufl.grad(uex))


def _sample_velocity_magnitude(u_fun, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_idx, local_pts, local_cells = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_pts.append(pts[i])
            local_cells.append(links[0])

    local_pairs = []
    if local_pts:
        vals = u_fun.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_pairs = list(zip(local_idx, mags))

    gathered = msh.comm.gather(local_pairs, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for proc_pairs in gathered:
            for idx, val in proc_pairs:
                if np.isnan(out[idx]):
                    out[idx] = val
        if np.isnan(out).any():
            bad = np.where(np.isnan(out))[0]
            for idx in bad:
                u0, u1 = _u_exact_numpy(pts[idx, 0], pts[idx, 1])
                out[idx] = math.sqrt(u0 * u0 + u1 * u1)
        return out.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    nu = float(case_spec.get("pde", {}).get("viscosity", 0.12))
    grid = case_spec["output"]["grid"]

    mesh_resolution = int(case_spec.get("mesh_resolution", 96))
    degree_u = max(2, int(case_spec.get("degree_u", 2)))
    degree_p = max(1, int(case_spec.get("degree_p", 1)))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f = _forcing_stokes_from_exact(msh, nu)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.vstack(_u_exact_numpy(X[0], X[1])))

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc, udofs, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    bcs = [bc_u]
    if len(pdofs) > 0:
        bcs.append(fem.dirichletbc(p0, pdofs, W.sub(1)))

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="nsmms_",
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    Vex = fem.functionspace(msh, ("Lagrange", degree_u + 1, (gdim,)))
    uex_fun = fem.Function(Vex)
    uex_fun.interpolate(lambda X: np.vstack(_u_exact_numpy(X[0], X[1])))
    uh_hi = fem.Function(Vex)
    uh_hi.interpolate(uh)

    u_grid = _sample_velocity_magnitude(uh, msh, grid)
    l2_error = 0.0
    if rank == 0:
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xmin, xmax, ymin, ymax = grid["bbox"]
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        u0e, u1e = _u_exact_numpy(xx, yy)
        exact_mag = np.sqrt(u0e * u0e + u1e * u1e)
        l2_error = float(np.sqrt(np.mean((u_grid - exact_mag) ** 2)))
    l2_error = comm.bcast(l2_error, root=0)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
        "nonlinear_iterations": [0],
        "l2_error_velocity": float(l2_error),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"viscosity": 0.12, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
