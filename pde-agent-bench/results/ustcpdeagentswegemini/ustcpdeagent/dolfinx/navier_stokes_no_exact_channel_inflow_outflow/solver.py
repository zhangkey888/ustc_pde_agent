import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl

ScalarType = PETSc.ScalarType

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
# linear_solver:        direct_lu
# preconditioner:       none
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```


def _inflow_profile(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    y = x[1]
    vals[0] = 4.0 * y * (1.0 - y)
    return vals


def _zero_vec(x):
    return np.zeros((2, x.shape[1]), dtype=np.float64)


def _build_spaces_and_bcs(mesh_resolution, degree_u=2, degree_p=1):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

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

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_zero, p_dofs, W.sub(1))
    return msh, W, V, Q, [bc_left, bc_bottom, bc_top, bc_p]


def _make_problem(msh, W, bcs, nu_value, advect_scale, w):
    v, q = ufl.TestFunctions(W)
    u, p = ufl.split(w)
    nu = fem.Constant(msh, ScalarType(nu_value))
    beta = fem.Constant(msh, ScalarType(advect_scale))
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=np.float64))

    def eps(a):
        return ufl.sym(ufl.grad(a))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + beta * ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    J = ufl.derivative(F, w)
    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1.0e-9,
            "snes_atol": 1.0e-10,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    return problem


def _sample_points(u_fun, msh, pts):
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    eval_pts = []
    eval_cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)
    if eval_pts:
        vals = u_fun.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    send = local_vals.copy()
    send[np.isnan(send)] = -1.0e300
    recv = np.empty_like(send)
    msh.comm.Allreduce(send, recv, op=MPI.MAX)
    recv[recv < -1.0e250] = np.nan
    return recv


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_points(u_fun, msh, pts)

    if np.isnan(vals).any():
        valid = np.where(~np.isnan(vals[:, 0]))[0]
        if valid.size == 0:
            vals[:] = 0.0
        else:
            nan_ids = np.where(np.isnan(vals[:, 0]))[0]
            for j in nan_ids:
                nearest = valid[np.argmin(np.abs(valid - j))]
                vals[j] = vals[nearest]

    return np.linalg.norm(vals, axis=1).reshape(ny, nx)


def _verification_metrics(u_fun, msh):
    ys = np.linspace(0.0, 1.0, 257)
    pts = np.column_stack([np.full_like(ys, 1.0), ys, np.zeros_like(ys)])
    vals = _sample_points(u_fun, msh, pts)
    valid = ~np.isnan(vals[:, 0])

    if np.any(valid):
        ux = vals[valid, 0]
        yy = ys[valid]
        exact = 4.0 * yy * (1.0 - yy)
        l2_error = float(np.sqrt(np.trapezoid((ux - exact) ** 2, yy)))
        flow_out = float(np.trapezoid(ux, yy))
    else:
        l2_error = float("inf")
        flow_out = float("nan")

    return {
        "poiseuille_outlet_l2": l2_error,
        "mass_flow_error": float(abs(flow_out - 2.0 / 3.0)) if np.isfinite(flow_out) else float("inf"),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    output = case_spec.get("output", {}).get("grid", {})
    nx = int(output.get("nx", 64))
    ny = int(output.get("ny", 64))
    bbox = output.get("bbox", [0.0, 1.0, 0.0, 1.0])

    pde = case_spec.get("pde", {})
    nu_value = float(pde.get("nu", 0.12)) if isinstance(pde, dict) else 0.12

    mesh_resolution = 96 if MPI.COMM_WORLD.size == 1 else 64
    degree_u = 2
    degree_p = 1

    msh, W, V, Q, bcs = _build_spaces_and_bcs(mesh_resolution, degree_u, degree_p)
    w = fem.Function(W)
    w.x.array[:] = 0.0
    w.sub(0).interpolate(_inflow_profile)
    w.x.scatter_forward()

    nonlinear_iterations = []
    linear_iterations_total = 0

    for beta in (0.0, 0.5, 1.0):
        problem = _make_problem(msh, W, bcs, nu_value, beta, w)
        problem.solve()
        w.x.scatter_forward()
        snes = problem.solver
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        try:
            linear_iterations_total += int(snes.getLinearSolveIterations())
        except Exception:
            pass

    u_fun = w.sub(0).collapse()
    u_grid = _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox)
    verification = _verification_metrics(u_fun, msh)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-9,
        "iterations": int(linear_iterations_total),
        "nonlinear_iterations": nonlinear_iterations,
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.12},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    result = solve(case_spec)
    wall = time.time() - t0
    l2_error = float(result["solver_info"]["verification"]["poiseuille_outlet_l2"])
    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", l2_error)
        print("WALL_TIME:", wall)
        print("GRID_SHAPE:", result["u"].shape)
        print("SOLVER_INFO:", result["solver_info"])
