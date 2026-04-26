import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _build_exact_fields(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector(
        (
            2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
            -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]),
        )
    )
    p_exact = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    grad_u = ufl.grad(u_exact)
    lap_u = ufl.as_vector([ufl.div(grad_u[i, :]) for i in range(msh.geometry.dim)])
    f = grad_u * u_exact - nu * lap_u + ufl.grad(p_exact)
    return u_exact, p_exact, f


def _interp_function(space, expr):
    fun = fem.Function(space)
    fun.interpolate(fem.Expression(expr, space.element.interpolation_points))
    return fun


def _sample_velocity_magnitude(u_fun, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_vals[np.array(idx_map, dtype=np.int32)] = mags

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            bad = np.where(np.isnan(out))[0][:10]
            raise RuntimeError(f"Failed to evaluate all output points, sample bad indices: {bad}")
        return out.reshape((ny, nx))
    return None


def _compute_errors(msh, u_h, p_h, u_exact_expr, p_exact_expr):
    Vv = u_h.function_space
    Qp = p_h.function_space

    u_ex = _interp_function(Vv, u_exact_expr)
    p_ex = _interp_function(Qp, p_exact_expr)

    err_u = fem.assemble_scalar(fem.form(ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx))
    ref_u = fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * ufl.dx))
    err_p = fem.assemble_scalar(fem.form((p_h - p_ex) * (p_h - p_ex) * ufl.dx))
    ref_p = fem.assemble_scalar(fem.form(p_ex * p_ex * ufl.dx))
    div_u = fem.assemble_scalar(fem.form(ufl.div(u_h) * ufl.div(u_h) * ufl.dx))

    comm = msh.comm
    err_u = comm.allreduce(err_u, op=MPI.SUM)
    ref_u = comm.allreduce(ref_u, op=MPI.SUM)
    err_p = comm.allreduce(err_p, op=MPI.SUM)
    ref_p = comm.allreduce(ref_p, op=MPI.SUM)
    div_u = comm.allreduce(div_u, op=MPI.SUM)

    return {
        "velocity_relative_l2_error": float(np.sqrt(err_u / max(ref_u, 1.0e-30))),
        "pressure_relative_l2_error": float(np.sqrt(err_p / max(ref_p, 1.0e-30))),
        "divergence_l2": float(np.sqrt(max(div_u, 0.0))),
    }


def solve(case_spec: dict) -> dict:
    """
    Return a dict with sampled velocity magnitude grid and solver metadata.
    """

    # DIAGNOSIS
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
    # special_notes: manufactured_solution

    # METHOD
    # spatial_method: fem
    # element_or_basis: Taylor-Hood_P2P1
    # stabilization: none
    # time_method: none
    # nonlinear_solver: newton
    # linear_solver: gmres
    # preconditioner: ilu
    # special_treatment: pressure_pinning
    # pde_skill: navier_stokes

    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("physics", {}).get("viscosity", 0.2))
    if "pde" in case_spec:
        nu = float(case_spec.get("pde", {}).get("nu", nu))

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # Use a moderately fine mesh/TH element for strong accuracy within the large time budget.
    mesh_resolution = int(case_spec.get("mesh_resolution", 40))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", 1))
    newton_rtol = float(case_spec.get("newton_rtol", 1.0e-9))
    newton_max_it = int(case_spec.get("newton_max_it", 25))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u_exact_expr, p_exact_expr, f_expr = _build_exact_fields(msh, nu)

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    u_bc_fun = _interp_function(V, u_exact_expr)
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc_fun, udofs, W.sub(0))

    p_pin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_pin_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    def eps(uu):
        return ufl.sym(ufl.grad(uu))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Initial guess from manufactured field helps robust Newton convergence.
    u_init, p_init = w.sub(0), w.sub(1)
    u_init.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    p_init.interpolate(fem.Expression(p_exact_expr, Q.element.interpolation_points))
    w.x.scatter_forward()

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1.0e-11,
        "snes_max_it": newton_max_it,
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-9,
        "pc_type": "ilu",
    }

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=opts
    )
    w = problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    errors = _compute_errors(msh, u_h, p_h, u_exact_expr, p_exact_expr)
    u_grid = _sample_velocity_magnitude(u_h, msh, nx_out, ny_out, bbox)

    wall = time.perf_counter() - t0
    nonlinear_its = [0]
    linear_iterations = 0
    try:
        snes = problem.solver
        nonlinear_its = [int(snes.getIterationNumber())]
        try:
            linear_iterations = int(snes.getLinearSolveIterations())
        except Exception:
            linear_iterations = 0
    except Exception:
        pass

    result = None
    if comm.rank == 0:
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": degree_u,
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "rtol": 1.0e-9,
                "iterations": linear_iterations,
                "nonlinear_iterations": nonlinear_its,
                "verification_velocity_relative_l2_error": errors["velocity_relative_l2_error"],
                "verification_pressure_relative_l2_error": errors["pressure_relative_l2_error"],
                "verification_divergence_l2": errors["divergence_l2"],
                "wall_time_sec": wall,
            },
        }
    return result if comm.rank == 0 else {"u": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 32,
                "ny": 32,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "physics": {"viscosity": 0.2},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
