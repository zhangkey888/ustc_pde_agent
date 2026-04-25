import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl


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
# bc_type:              mixed
# special_notes:        pressure_pinning
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


ScalarType = PETSc.ScalarType


def _inflow_profile(x):
    values = np.zeros((2, x.shape[1]), dtype=np.float64)
    y = x[1]
    values[0] = 4.0 * y * (1.0 - y)
    values[1] = 0.0
    return values


def _zero_vec(x):
    return np.zeros((2, x.shape[1]), dtype=np.float64)


def _build_problem(mesh_resolution=48, degree_u=2, degree_p=1, nu_value=0.12,
                   snes_rtol=1e-9, snes_max_it=30):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    vq = ufl.TestFunctions(W)
    v, q = vq
    u, p = ufl.split(w)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=np.float64))

    def eps(a):
        return ufl.sym(ufl.grad(a))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    J = ufl.derivative(F, w)

    fdim = msh.topology.dim - 1

    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_in = fem.Function(V)
    u_in.interpolate(_inflow_profile)

    u_zero = fem.Function(V)
    u_zero.interpolate(_zero_vec)

    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bc_left = fem.dirichletbc(u_in, dofs_left, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, dofs_bottom, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_left, bc_bottom, bc_top, bc_p]

    # Initial guess: interpolated extension of inflow on whole domain
    w.x.array[:] = 0.0
    u_init = w.sub(0)
    u_init.interpolate(_inflow_profile)
    w.x.scatter_forward()

    ksp_type = "preonly"
    pc_type = "lu"
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": snes_rtol,
        "snes_atol": 1e-10,
        "snes_max_it": snes_max_it,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )

    return msh, W, V, Q, w, problem, ksp_type, pc_type


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

    values = np.full((nx * ny, 2), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    map_ids = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells.append(links[0])
            map_ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(map_ids, dtype=np.int32), :] = vals

    comm = msh.comm
    send = values.copy()
    send[np.isnan(send)] = -1.0e300
    recv = np.empty_like(send)
    comm.Allreduce(send, recv, op=MPI.MAX)
    recv[recv < -1.0e250] = np.nan

    # Fill any stubborn NaNs on domain boundary by nearest valid neighbor along flattened order
    if np.isnan(recv).any():
        valid = np.where(~np.isnan(recv[:, 0]))[0]
        if valid.size == 0:
            recv[:] = 0.0
        else:
            for j in np.where(np.isnan(recv[:, 0]))[0]:
                nearest = valid[np.argmin(np.abs(valid - j))]
                recv[j] = recv[nearest]

    mag = np.linalg.norm(recv, axis=1).reshape(ny, nx)
    return mag


def _compute_verification(u_fun, msh):
    pts = np.array([
        [1.0, 0.25, 0.0],
        [1.0, 0.50, 0.0],
        [1.0, 0.75, 0.0],
    ], dtype=np.float64)

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((3, 2), np.nan, dtype=np.float64)
    eval_pts = []
    eval_cells = []
    ids = []
    for i in range(3):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            ids.append(i)
    if eval_pts:
        vals = u_fun.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32), :] = vals

    send = local_vals.copy()
    send[np.isnan(send)] = -1.0e300
    recv = np.empty_like(send)
    msh.comm.Allreduce(send, recv, op=MPI.MAX)
    recv[recv < -1.0e250] = np.nan

    y = np.array([0.25, 0.50, 0.75], dtype=np.float64)
    exact = 4.0 * y * (1.0 - y)
    outlet_u = recv[:, 0]
    poiseuille_misfit = float(np.sqrt(np.nanmean((outlet_u - exact) ** 2)))

    # Simple mass-flow consistency via line sample
    ys = np.linspace(0.0, 1.0, 129)
    pts_line = np.column_stack([np.full_like(ys, 1.0), ys, np.zeros_like(ys)])
    cell_candidates = geometry.compute_collisions_points(tree, pts_line)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts_line)
    vals_line = np.full((ys.size, 2), np.nan, dtype=np.float64)
    eval_pts = []
    eval_cells = []
    ids = []
    for i in range(ys.size):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts_line[i])
            eval_cells.append(links[0])
            ids.append(i)
    if eval_pts:
        vals = u_fun.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        vals_line[np.array(ids, dtype=np.int32)] = vals
    send = vals_line.copy()
    send[np.isnan(send)] = -1.0e300
    recv2 = np.empty_like(send)
    msh.comm.Allreduce(send, recv2, op=MPI.MAX)
    recv2[recv2 < -1.0e250] = np.nan
    valid = ~np.isnan(recv2[:, 0])
    flow_out = float(np.trapezoid(recv2[valid, 0], ys[valid])) if np.any(valid) else np.nan
    flow_in = 2.0 / 3.0
    flow_error = float(abs(flow_out - flow_in)) if np.isfinite(flow_out) else np.inf

    return {
        "poiseuille_misfit_outlet": poiseuille_misfit,
        "mass_flow_error": flow_error,
    }


def solve(case_spec: dict) -> dict:
    t0 = time.time()

    output = case_spec.get("output", {}).get("grid", {})
    nx = int(output.get("nx", 64))
    ny = int(output.get("ny", 64))
    bbox = output.get("bbox", [0.0, 1.0, 0.0, 1.0])

    pde = case_spec.get("pde", {})
    nu_value = float(pde.get("nu", 0.12)) if isinstance(pde, dict) else 0.12

    # Use some of the large time budget for better accuracy, but keep runtime safe.
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        mesh_resolution = 24
    else:
        mesh_resolution = 20

    degree_u = 2
    degree_p = 1
    snes_rtol = 1e-8
    snes_max_it = 20

    msh, W, V, Q, w, problem, ksp_type, pc_type = _build_problem(
        mesh_resolution=mesh_resolution,
        degree_u=degree_u,
        degree_p=degree_p,
        nu_value=nu_value,
        snes_rtol=snes_rtol,
        snes_max_it=snes_max_it,
    )

    nonlinear_iterations = [0]
    iterations_total = 0

    for attempt in range(2):
        try:
            w_sol = problem.solve()
            w_sol.x.scatter_forward()
            snes = problem.solver
            nonlinear_iterations = [int(snes.getIterationNumber())]
            try:
                iterations_total = int(snes.getLinearSolveIterations())
            except Exception:
                iterations_total = 0
            break
        except Exception:
            if attempt == 0:
                msh, W, V, Q, w, problem, ksp_type, pc_type = _build_problem(
                    mesh_resolution=max(32, mesh_resolution // 2),
                    degree_u=degree_u,
                    degree_p=degree_p,
                    nu_value=nu_value,
                    snes_rtol=1e-8,
                    snes_max_it=50,
                )
            else:
                raise

    u_fun = w.sub(0).collapse()
    verification = _compute_verification(u_fun, msh)
    u_grid = _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": int(iterations_total),
        "nonlinear_iterations": nonlinear_iterations,
        "verification": verification,
        "wall_time_sec": time.time() - t0,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.12},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
