import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type: stokes
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector+scalar
# coupling: saddle_point
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: mixed
# special_notes: pressure_pinning
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: minres
# preconditioner: hypre
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```


def _inlet_profile(x):
    values = np.zeros((2, x.shape[1]), dtype=np.float64)
    y = x[1]
    values[0] = 4.0 * y * (1.0 - y)
    values[1] = 0.0
    return values


def _zero_vec(x):
    return np.zeros((2, x.shape[1]), dtype=np.float64)


def _build_spaces(msh):
    gdim = msh.geometry.dim
    cell = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell, 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1

    inlet_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    inlet_fun = fem.Function(V)
    inlet_fun.interpolate(_inlet_profile)

    zero_fun = fem.Function(V)
    zero_fun.interpolate(_zero_vec)

    inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_inlet = fem.dirichletbc(inlet_fun, inlet_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(zero_fun, bottom_dofs, W.sub(0))
    bc_top = fem.dirichletbc(zero_fun, top_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    return [bc_inlet, bc_bottom, bc_top, bc_p]


def _solve_stokes_once(mesh_resolution, nu, ksp_type="minres", pc_type="hypre", rtol=1.0e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh)
    bcs = _build_bcs(msh, W, V, Q)

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    f = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=np.float64))

    a = (
        2.0 * ScalarType(nu) * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    used_ksp = ksp_type
    used_pc = pc_type
    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if ksp_type in ("minres", "gmres"):
        opts["ksp_max_it"] = 10000

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"stokes_{mesh_resolution}_",
            petsc_options=opts,
        )
        wh = problem.solve()
    except Exception:
        used_ksp = "preonly"
        used_pc = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"stokes_lu_{mesh_resolution}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        wh = problem.solve()

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    verification = _verification_metrics(msh, uh, ph, nu)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification": verification,
    }
    return msh, uh, ph, solver_info


def _eval_function(msh, func, points):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    pts_local = []
    cells_local = []
    map_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            map_local.append(i)

    value_shape = func.function_space.element.value_shape
    value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    out = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    if pts_local:
        vals = func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        vals = np.real(vals)
        vals = np.asarray(vals, dtype=np.float64).reshape(len(pts_local), value_size)
        out[np.asarray(map_local, dtype=np.int32), :] = vals
    return out


def _verification_metrics(msh, uh, ph, nu):
    xs = np.linspace(0.1, 0.9, 9)
    ys = np.linspace(0.0, 1.0, 101)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    vals = _eval_function(msh, uh, pts)
    ux = vals[:, 0].reshape(len(ys), len(xs))
    uy = vals[:, 1].reshape(len(ys), len(xs))

    poiseuille = (4.0 * Y * (1.0 - Y)).reshape(len(ys), len(xs))
    rel_profile_err = float(np.linalg.norm(ux - poiseuille) / max(np.linalg.norm(poiseuille), 1.0e-14))
    transverse_rel = float(np.linalg.norm(uy) / max(np.linalg.norm(ux), 1.0e-14))

    try:
        div_l2 = float(np.sqrt(msh.comm.allreduce(
            fem.assemble_scalar(fem.form((ufl.div(uh) ** 2) * ufl.dx)), op=MPI.SUM
        )))
    except Exception:
        div_l2 = float("nan")

    speed_sq = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))
    speed_sq = msh.comm.allreduce(speed_sq, op=MPI.SUM)
    speed_l2 = float(np.sqrt(max(speed_sq, 0.0)))

    p_mean = fem.assemble_scalar(fem.form(ph * ufl.dx))
    p_mean = msh.comm.allreduce(p_mean, op=MPI.SUM)
    vol = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh)))
    vol = msh.comm.allreduce(vol, op=MPI.SUM)

    return {
        "poiseuille_relative_profile_error": rel_profile_err,
        "transverse_velocity_relative_norm": transverse_rel,
        "divergence_l2": div_l2,
        "velocity_l2": speed_l2,
        "pressure_mean": float(p_mean / max(vol, 1.0e-14)),
        "viscosity": float(nu),
    }


def _sample_velocity_magnitude(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])
    vals = _eval_function(msh, uh, points)
    mag = np.linalg.norm(vals[:, :2], axis=1)
    mag = np.where(np.isnan(mag), 0.0, mag)
    return mag.reshape(ny, nx)


def _estimate_grid_convergence(coarse_grid, fine_grid):
    return float(np.linalg.norm(fine_grid - coarse_grid) / max(np.linalg.norm(fine_grid), 1.0e-14))


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    nu = float(pde.get("nu", case_spec.get("nu", 0.05)))
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])

    time_limit = float(case_spec.get("time_limit", case_spec.get("wall_time_limit", 3717.917)))
    start = time.time()

    base_res = max(24, min(96, 2 * max(nx_out, ny_out)))
    candidates = [base_res]
    if time_limit > 20:
        candidates.append(min(128, max(base_res + 16, int(1.4 * base_res))))
    if time_limit > 120:
        candidates.append(min(160, max(base_res + 32, int(1.8 * base_res))))
    candidates = sorted(set(candidates))

    best = None
    previous_grid = None
    previous_info = None

    for res in candidates:
        msh, uh, ph, info = _solve_stokes_once(res, nu)
        u_grid = _sample_velocity_magnitude(msh, uh, grid)

        if previous_grid is not None:
            info["verification"]["grid_convergence_relative_change"] = _estimate_grid_convergence(previous_grid, u_grid)
        else:
            info["verification"]["grid_convergence_relative_change"] = None

        best = (u_grid, info)
        previous_grid = u_grid
        previous_info = info

        elapsed = time.time() - start
        if elapsed > 0.5 * time_limit:
            break

    u_grid, solver_info = best
    solver_info["wall_time_sec"] = float(time.time() - start)
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.05, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 60.0,
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
