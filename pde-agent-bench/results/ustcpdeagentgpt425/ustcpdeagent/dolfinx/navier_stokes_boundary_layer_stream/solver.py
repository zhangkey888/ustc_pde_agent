import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


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
# solution_regularity:  boundary_layer
# bc_type:              all_dirichlet
# special_notes:        pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P2P1
# stabilization:        none
# time_method:          none
# nonlinear_solver:     newton
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```


def _manufactured_fields(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact = ufl.as_vector(
        [
            pi * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.cos(pi * x[1]),
            -6.0 * ufl.exp(6.0 * (x[0] - 1.0)) * ufl.sin(pi * x[1]),
        ]
    )
    p_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    f_expr = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    return u_exact, p_exact, f_expr


def _all_boundary_facets(msh):
    fdim = msh.topology.dim - 1
    return mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))


def _sample_velocity_magnitude(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    eps = 1.0e-12
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xs = np.clip(xs, bbox[0] + eps, bbox[1] - eps)
    ys = np.clip(ys, bbox[2] + eps, bbox[3] - eps)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    point_ids = []
    points_local = []
    cells_local = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            points_local.append(pts[i])
            cells_local.append(links[0])

    vals_full = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    if len(points_local) > 0:
        vals = u_func.eval(np.array(points_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals_full[np.array(point_ids, dtype=np.int32), :] = vals

    gathered = msh.comm.gather(vals_full, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(vals_full, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            for j in range(merged.shape[1]):
                col = merged[:, j]
                col[np.isnan(col)] = 0.0
                merged[:, j] = col
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
        return mag

    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    nu = 0.08
    if "pde" in case_spec and isinstance(case_spec["pde"], dict):
        nu = float(case_spec["pde"].get("nu", case_spec["pde"].get("viscosity", nu)))

    time_limit = None
    if isinstance(case_spec, dict):
        for key in ("time_limit", "wall_time_sec", "max_wall_time_sec"):
            if key in case_spec:
                try:
                    time_limit = float(case_spec[key])
                    break
                except Exception:
                    pass

    mesh_resolution = 56
    element_degree = 2
    if time_limit is not None and time_limit > 120:
        mesh_resolution = 72
    if time_limit is not None and time_limit > 300:
        mesh_resolution = 88

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u_exact_ufl, p_exact_ufl_raw, f_ufl = _manufactured_fields(msh, nu)
    x = ufl.SpatialCoordinate(msh)
    p_exact_ufl = p_exact_ufl_raw - ufl.sin(ufl.pi * 0.0) * ufl.sin(ufl.pi * 0.0)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    bfacets = _all_boundary_facets(msh)
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc, udofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    w_stokes = fem.Function(W)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a_stokes = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_stokes = ufl.inner(f_fun, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "gmres",
            "ksp_rtol": 1e-9,
            "pc_type": "lu",
        },
    )
    w_stokes = stokes_problem.solve()
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    (u_nl, p_nl) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u_nl), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_nl) * u_nl, v) * ufl.dx
        - ufl.inner(p_nl, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u_nl), q) * ufl.dx
        - ufl.inner(f_fun, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_iterations = []
    chosen_ksp = "gmres"
    chosen_pc = "ilu"
    chosen_rtol = 1e-9
    total_linear_iterations = 0

    snes_monitor = []

    def _monitor(snes, its, norm):
        snes_monitor.append((its, norm))

    try:
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-10,
                "snes_max_it": 30,
                "ksp_type": "gmres",
                "ksp_rtol": chosen_rtol,
                "pc_type": "ilu",
            },
        )
        problem.solver.setMonitor(_monitor)
        w = problem.solve()
    except Exception:
        chosen_pc = "lu"
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_fallback_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-10,
                "snes_max_it": 35,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        problem.solver.setMonitor(_monitor)
        w = problem.solve()
        chosen_ksp = "preonly"

    try:
        nit = int(problem.solver.getIterationNumber())
        nonlinear_iterations = [max(1, nit)]
        ksp = problem.solver.getKSP()
        total_linear_iterations = int(ksp.getIterationNumber())
    except Exception:
        nonlinear_iterations = [max(1, len(snes_monitor) - 1)]
        total_linear_iterations = 0

    w.x.scatter_forward()
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    u_chk, p_chk = ufl.split(w)

    err_u_form = fem.form(ufl.inner(u_chk - u_exact_ufl, u_chk - u_exact_ufl) * ufl.dx)
    div_u_form = fem.form(ufl.inner(ufl.div(u_chk), ufl.div(u_chk)) * ufl.dx)

    err_u_sq = comm.allreduce(fem.assemble_scalar(err_u_form), op=MPI.SUM)
    div_u_sq = comm.allreduce(fem.assemble_scalar(div_u_form), op=MPI.SUM)

    p_arr = np.nan_to_num(p_h.x.array, nan=0.0, posinf=0.0, neginf=0.0)
    p_local_min = np.min(p_arr) if p_arr.size > 0 else 0.0
    p_local_max = np.max(p_arr) if p_arr.size > 0 else 0.0
    p_min = comm.allreduce(float(p_local_min), op=MPI.MIN)
    p_max = comm.allreduce(float(p_local_max), op=MPI.MAX)

    err_u_local = np.sqrt(max(0.0, float(err_u_sq)))
    err_p_local = float(p_max - p_min) if np.isfinite(p_max) and np.isfinite(p_min) else 0.0
    div_u_local = np.sqrt(max(0.0, float(div_u_sq)))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    u_grid = _sample_velocity_magnitude(u_h, bbox, nx, ny)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(chosen_ksp),
        "pc_type": str(chosen_pc),
        "rtol": float(chosen_rtol),
        "iterations": int(total_linear_iterations),
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "verification": {
            "velocity_l2_error": float(err_u_local),
            "pressure_range": float(err_p_local),
            "divergence_l2_error": float(div_u_local),
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }

    if rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "navier_stokes", "time": False, "viscosity": 0.08},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
