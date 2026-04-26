import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


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
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Taylor-Hood_P2P1
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: pressure_pinning
# pde_skill: stokes
# ```


ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _exact_velocity_expr(x):
    return ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _exact_pressure_expr(x):
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def _forcing_expr(msh, nu_value):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _exact_velocity_expr(x)
    p_ex = _exact_pressure_expr(x)
    return -nu_value * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)


def _build_solver(mesh_resolution=96):
    msh = mesh.create_unit_square(
        COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    nu = ScalarType(1.0)
    x = ufl.SpatialCoordinate(msh)
    u_ex = _exact_velocity_expr(x)
    p_ex = _exact_pressure_expr(x)
    f_expr = _forcing_expr(msh, nu)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_rtol": 1.0e-12,
        "pc_factor_mat_solver_type": "mumps",
    }

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_basic_",
            petsc_options=opts,
        )
        wh = problem.solve()
    except Exception:
        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-12,
        }
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_basic_fallback_",
            petsc_options=opts,
        )
        wh = problem.solve()

    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    u_exact_fun.x.scatter_forward()

    diff = fem.Function(V)
    diff.x.array[:] = uh.x.array - u_exact_fun.x.array
    diff.x.scatter_forward()
    discrete_max_err = np.max(np.abs(diff.x.array)) if diff.x.array.size else 0.0
    global_max_err = COMM.allreduce(discrete_max_err, op=MPI.MAX)

    err_form = fem.form(ufl.inner(uh - u_exact_fun, uh - u_exact_fun) * ufl.dx)
    l2_sq = fem.assemble_scalar(err_form)
    l2_sq = COMM.allreduce(l2_sq, op=MPI.SUM)
    l2_err = math.sqrt(max(l2_sq, 0.0))

    return msh, uh, ph, {
        "mesh_resolution": mesh_resolution,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 1,
        "verification": {
            "velocity_L2_error": float(l2_err),
            "velocity_dof_max_error": float(global_max_err),
        },
    }


def _eval_function_on_points(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    owned_pts = []
    owned_cells = []
    owned_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            owned_pts.append(points[i])
            owned_cells.append(links[0])
            owned_ids.append(i)

    vals_local = None
    if owned_pts:
        vals_local = func.eval(np.array(owned_pts, dtype=np.float64), np.array(owned_cells, dtype=np.int32))
    return owned_ids, vals_local


def _sample_velocity_magnitude(uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    owned_ids, vals_local = _eval_function_on_points(uh, pts)

    gathered_ids = COMM.gather(np.array(owned_ids, dtype=np.int64), root=0)
    gathered_vals = COMM.gather(vals_local, root=0)

    if COMM.rank == 0:
        full = np.full((nx * ny, uh.function_space.mesh.geometry.dim), np.nan, dtype=np.float64)
        for ids, vals in zip(gathered_ids, gathered_vals):
            if ids.size > 0 and vals is not None:
                full[ids] = np.asarray(vals, dtype=np.float64)
        if np.isnan(full).any():
            raise RuntimeError("Point sampling failed for some grid points.")
        mag = np.linalg.norm(full, axis=1).reshape(ny, nx)
        return mag
    return None


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    time_limit = 9.773
    # Adaptive time-accuracy trade-off: choose the finest mesh likely under time budget.
    candidate_resolutions = [72, 96, 112, 128]
    chosen = candidate_resolutions[0]

    if COMM.rank == 0:
        # Conservative default without profiling solves in solve().
        # For smooth manufactured Stokes with direct LU in 2D, 96 is usually a strong balance.
        chosen = 96
    chosen = COMM.bcast(chosen, root=0)

    msh, uh, ph, solver_info = _build_solver(mesh_resolution=chosen)

    u_grid = _sample_velocity_magnitude(uh, case_spec["output"]["grid"])

    elapsed = time.perf_counter() - t0
    solver_info["wall_time_sec"] = float(elapsed)
    solver_info["time_budget_sec"] = float(time_limit)

    if COMM.rank == 0:
        # Additional verification on output grid against exact velocity magnitude
        grid_spec = case_spec["output"]["grid"]
        nx = int(grid_spec["nx"])
        ny = int(grid_spec["ny"])
        bbox = grid_spec["bbox"]
        xmin, xmax, ymin, ymax = map(float, bbox)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u1 = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
        u2 = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
        mag_ex = np.sqrt(u1 * u1 + u2 * u2)
        solver_info["verification"]["output_grid_max_error"] = float(np.max(np.abs(u_grid - mag_ex)))
        solver_info["verification"]["output_grid_l2_mean_error"] = float(np.sqrt(np.mean((u_grid - mag_ex) ** 2)))

        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
