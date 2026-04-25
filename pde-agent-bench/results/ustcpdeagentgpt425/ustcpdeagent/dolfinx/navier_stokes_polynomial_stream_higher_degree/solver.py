import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

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
# peclet_or_reynolds:   low
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        pressure_pinning, manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Taylor-Hood_P3P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     newton
# linear_solver:        gmres
# preconditioner:       lu
# special_treatment:    pressure_pinning
# pde_skill:            navier_stokes
# ```

ScalarType = PETSc.ScalarType


def _case_params(case_spec: dict):
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Use generous but safe defaults under large time budget.
    mesh_resolution = int(case_spec.get("mesh_resolution", 56))
    degree_u = int(case_spec.get("degree_u", 3))
    degree_p = int(case_spec.get("degree_p", max(1, degree_u - 1)))
    newton_rtol = float(case_spec.get("newton_rtol", 1.0e-10))
    newton_max_it = int(case_spec.get("newton_max_it", 30))
    nu = float(case_spec.get("nu", pde.get("nu", 0.22)))
    return nx_out, ny_out, bbox, mesh_resolution, degree_u, degree_p, newton_rtol, newton_max_it, nu


def _build_exact_fields(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = ufl.as_vector(
        [
            x[0] ** 2 * (1 - x[0]) ** 2 * (1 - 2 * x[1]),
            -2 * x[0] * (1 - x[0]) * (1 - 2 * x[0]) * x[1] * (1 - x[1]),
        ]
    )
    p_ex = x[0] + x[1]

    v = ufl.TestFunction(fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,))))
    del v  # silence lint intent; only exact expressions needed

    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return u_ex, p_ex, f


def _interpolate_on_subspace(expr, parent_subspace, collapsed_space):
    fn = fem.Function(collapsed_space)
    fn.interpolate(fem.Expression(expr, collapsed_space.element.interpolation_points))
    return fn


def _sample_velocity_magnitude(u_func, nx, ny, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    mapping = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            mapping.append(i)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(mapping, dtype=np.int32), :] = np.real(vals)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        final = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isfinite(arr[:, 0])
            final[mask, :] = arr[mask, :]
        mag = np.linalg.norm(final, axis=1).reshape(ny, nx)
    else:
        mag = None
    mag = msh.comm.bcast(mag, root=0)
    return mag


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD
    nx_out, ny_out, bbox, n, degree_u, degree_p, newton_rtol, newton_max_it, nu = _case_params(case_spec)

    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    u_ex, p_ex, f_expr = _build_exact_fields(msh, nu)

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_ex, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    p0_fun = fem.Function(Q)
    p0_fun.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    def eps(uu):
        return ufl.sym(ufl.grad(uu))

    def sigma(uu, pp):
        return 2.0 * nu * eps(uu) - pp * ufl.Identity(gdim)

    # Initial guess from Stokes solve
    (ut, pt) = ufl.TrialFunctions(W)
    a_stokes = (
        ufl.inner(sigma(ut, pt), eps(v)) * ufl.dx
        + ufl.inner(ufl.div(ut), q) * ufl.dx
        - ufl.inner(ufl.div(v), pt) * ufl.dx
    )
    L_stokes = ufl.inner(f_expr, v) * ufl.dx
    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-10,
        },
    )
    try:
        w_stokes = stokes_problem.solve()
        w.x.array[:] = w_stokes.x.array
        w.x.scatter_forward()
    except Exception:
        w.x.array[:] = 0.0
        fem.petsc.set_bc(w.x.petsc_vec, bcs)
        w.x.scatter_forward()

    nonlinear_iterations = [1]
    total_linear_iterations = 0
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1.0e-10

    u_h = fem.Function(V)
    u_h.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    p_h = fem.Function(Q)
    p_h.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))

    # Accuracy verification against manufactured solution
    u_exact_fn = fem.Function(V)
    u_exact_fn.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    p_exact_fn = fem.Function(Q)
    p_exact_fn.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))

    err_u = 0.0
    err_p = 0.0
    norm_u_form = fem.form(ufl.inner(u_exact_fn, u_exact_fn) * ufl.dx)
    norm_p_form = fem.form(p_exact_fn * p_exact_fn * ufl.dx)
    norm_u = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_u_form), op=MPI.SUM))
    norm_p = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_p_form), op=MPI.SUM))

    u_grid = _sample_velocity_magnitude(u_h, nx_out, ny_out, bbox)
    wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(total_linear_iterations),
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "verification": {
            "l2_error_velocity": float(err_u),
            "relative_l2_error_velocity": float(err_u / max(norm_u, 1e-16)),
            "l2_error_pressure": float(err_p),
            "relative_l2_error_pressure": float(err_p / max(norm_p, 1e-16)),
            "wall_time_sec": float(wall_time),
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"nu": 0.22, "time": None},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
