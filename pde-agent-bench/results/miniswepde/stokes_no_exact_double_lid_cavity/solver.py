import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve Stokes flow (incompressible) on a unit square domain.
    Double lid cavity: top and bottom walls move.
    """
    comm = MPI.COMM_WORLD

    pde_spec = case_spec.get("pde", {})
    nu_val = float(pde_spec.get("viscosity", 0.3))

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 100)
    ny_out = output.get("ny", 100)

    bc_spec = pde_spec.get("boundary_conditions", [])

    # Adaptive mesh refinement
    resolutions = [32, 64, 96]
    prev_norm = None
    final_result = None
    final_info = None

    for N in resolutions:
        result, info, cur_norm = _solve_stokes(comm, N, nu_val, bc_spec, nx_out, ny_out)

        if prev_norm is not None:
            rel_change = abs(cur_norm - prev_norm) / (abs(cur_norm) + 1e-15)
            if rel_change < 0.005:
                return {"u": result, "solver_info": info}

        prev_norm = cur_norm
        final_result = result
        final_info = info

    return {"u": final_result, "solver_info": final_info}


def _solve_stokes(comm, N, nu_val, bc_spec, nx_out, ny_out):
    """Solve Stokes on mesh of resolution N."""

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    degree_u = 2
    degree_p = 1

    P2 = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    P1_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    ME = basix.ufl.mixed_element([P2, P1_el])
    W = fem.functionspace(domain, ME)

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(domain, ScalarType((0.0, 0.0)))
    nu = fem.Constant(domain, ScalarType(nu_val))

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    # Build boundary conditions
    bcs = _build_bcs(domain, W, V, bc_spec, fdim, gdim)
    if len(bcs) == 0:
        bcs = _default_double_lid_bcs(domain, W, V, fdim, gdim)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol_val = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    u_sol = wh.sub(0).collapse()

    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out, gdim)

    norm_val = np.sqrt(np.nanmean(u_grid**2))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol_val,
        "iterations": 1,
    }

    return u_grid, solver_info, norm_val


def _build_bcs(domain, W, V, bc_spec, fdim, gdim):
    """Parse boundary conditions from case_spec."""
    bcs = []
    if not isinstance(bc_spec, list):
        return bcs

    for bc_item in bc_spec:
        if not isinstance(bc_item, dict):
            continue
        bc_type = bc_item.get("type", "dirichlet")
        if bc_type != "dirichlet":
            continue

        location = bc_item.get("location", "")
        value = bc_item.get("value", None)
        variable = bc_item.get("variable", "u")

        if variable not in ("u", "velocity"):
            continue

        if isinstance(value, list):
            val_x = float(value[0])
            val_y = float(value[1])
        else:
            val_x, val_y = 0.0, 0.0

        marker = _get_boundary_marker(location)
        if marker is not None:
            facets = mesh.locate_entities_boundary(domain, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            u_bc_func = fem.Function(V)
            vx, vy = float(val_x), float(val_y)
            u_bc_func.interpolate(lambda x, _vx=vx, _vy=vy: np.vstack([
                np.full(x.shape[1], _vx), np.full(x.shape[1], _vy)
            ]))
            bcs.append(fem.dirichletbc(u_bc_func, dofs, W.sub(0)))

    return bcs


def _default_double_lid_bcs(domain, W, V, fdim, gdim):
    """Default double-lid driven cavity BCs."""
    bcs = []

    # Top wall: u = (1, 0)
    top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))

    # Bottom wall: u = (-1, 0)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack([-np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0)))

    # Left wall: u = (0, 0)
    left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
    bcs.append(fem.dirichletbc(u_left, left_dofs, W.sub(0)))

    # Right wall: u = (0, 0)
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    right_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.zeros((gdim, x.shape[1])))
    bcs.append(fem.dirichletbc(u_right, right_dofs, W.sub(0)))

    return bcs


def _get_boundary_marker(location_str):
    """Convert location string to boundary marker function."""
    loc = str(location_str).lower().strip()
    if "top" in loc or "y=1" in loc or "y = 1" in loc:
        return lambda x: np.isclose(x[1], 1.0)
    elif "bottom" in loc or "y=0" in loc or "y = 0" in loc:
        return lambda x: np.isclose(x[1], 0.0)
    elif "left" in loc or "x=0" in loc or "x = 0" in loc:
        return lambda x: np.isclose(x[0], 0.0)
    elif "right" in loc or "x=1" in loc or "x = 1" in loc:
        return lambda x: np.isclose(x[0], 1.0)
    elif "all" in loc or "boundary" in loc:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    return None


def _evaluate_on_grid(domain, u_func, nx, ny, gdim):
    """Evaluate velocity magnitude on a uniform grid."""
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    n_points = points_3d.shape[0]
    u_values = np.full((n_points, gdim), np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        for idx, map_idx in enumerate(eval_map):
            u_values[map_idx, :] = vals[idx, :gdim]

    vel_mag = np.sqrt(np.sum(u_values**2, axis=1))
    vel_mag_grid = vel_mag.reshape((nx, ny))

    return vel_mag_grid
