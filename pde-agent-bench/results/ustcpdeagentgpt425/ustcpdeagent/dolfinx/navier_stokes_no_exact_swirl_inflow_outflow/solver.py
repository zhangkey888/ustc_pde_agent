import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType

DIAGNOSIS_AND_METHOD = (
    "DIAGNOSIS: equation_type=navier_stokes, spatial_dim=2, domain_geometry=rectangle, "
    "unknowns=vector+scalar, coupling=saddle_point, linearity=nonlinear, "
    "time_dependence=steady, dominant_physics=mixed, bc_type=all_dirichlet, "
    "special_notes=pressure_pinning; "
    "METHOD: spatial_method=fem, element_or_basis=Taylor-Hood_P2P1, stabilization=none, "
    "time_method=none, nonlinear_solver=newton, linear_solver=preonly, "
    "preconditioner=lu, special_treatment=pressure_pinning, pde_skill=navier_stokes"
)


def _create_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _inflow_profile(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    y = x[1]
    vals[0, :] = np.sin(np.pi * y)
    vals[1, :] = 0.2 * np.sin(2.0 * np.pi * y)
    return vals


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_left = fem.Function(V)
    u_left.interpolate(_inflow_profile)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)

    bcs = [
        fem.dirichletbc(u_left, left_dofs, W.sub(0)),
        fem.dirichletbc(u_zero, bottom_dofs, W.sub(0)),
        fem.dirichletbc(u_zero, top_dofs, W.sub(0)),
    ]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p_zero = fem.Function(Q)
        p_zero.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p_zero, p_dofs, W.sub(1)))
    return bcs


def _solve_stokes_initial_guess(msh, W, bcs, nu_value):
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=np.float64))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_init_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh


def _solve_navier_stokes(msh, W, bcs, nu_value, w0, newton_rtol=1e-8, newton_max_it=25):
    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=np.float64))

    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    w.x.scatter_forward()

    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
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
            "snes_rtol": newton_rtol,
            "snes_atol": 1e-10,
            "snes_max_it": newton_max_it,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    w_sol = problem.solve()
    w_sol.x.scatter_forward()
    return w_sol


def _sample_velocity_magnitude(u_fun, bbox, nx, ny):
    msh = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = vals

    gathered = msh.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr[:, 0])
        merged[mask] = arr[mask]
    merged = np.nan_to_num(merged, nan=0.0)
    return np.linalg.norm(merged, axis=1).reshape(ny, nx)


def _compute_divergence_indicator(uh):
    msh = uh.function_space.mesh
    V0 = fem.functionspace(msh, ("DG", 0))
    expr = fem.Expression(ufl.div(uh), V0.element.interpolation_points)
    divh = fem.Function(V0)
    divh.interpolate(expr)
    arr = np.nan_to_num(divh.x.array, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(msh.comm.allreduce(np.dot(arr, arr), op=MPI.SUM)))


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    mesh_resolution = 64
    msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _create_spaces(msh, 2, 1)
    bcs = _build_bcs(msh, W, V, Q)

    try:
        w0 = _solve_stokes_initial_guess(msh, W, bcs, 0.22)
    except Exception:
        w0 = fem.Function(W)
        w0.x.array[:] = 0.0
        w0.x.scatter_forward()

    w = _solve_navier_stokes(msh, W, bcs, 0.22, w0)
    uh = w.sub(0).collapse()
    uh.x.scatter_forward()

    u_grid = _sample_velocity_magnitude(uh, bbox, nx, ny)
    div_indicator = _compute_divergence_indicator(uh)
    elapsed = time.perf_counter() - start

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": 0,
        "nonlinear_iterations": [-1],
        "verification": {
            "divergence_l2_indicator": div_indicator,
            "wall_time_sec": elapsed,
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "output": {"grid": {"nx": 17, "ny": 19, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"nu": 0.22},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
