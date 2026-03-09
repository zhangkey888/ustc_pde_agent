import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    nu_val = float(pde.get("viscosity", 0.4))
    source = pde.get("source", ["1.0", "0.0"])
    f_vals = [float(s) for s in source]
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 100)
    ny_out = output.get("ny", 100)

    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    prev_norm = None
    final_result = None
    for N in resolutions:
        result = _solve_stokes(case_spec, N, nu_val, f_vals, x_range, y_range, nx_out, ny_out)
        current_norm = np.nanmean(np.abs(result["u"]))

        if prev_norm is not None:
            # If both norms are very small, consider converged
            if abs(current_norm) < 1e-12 and abs(prev_norm) < 1e-12:
                final_result = result
                break
            if abs(current_norm) > 1e-15:
                rel_err = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
                if rel_err < 0.005:
                    final_result = result
                    break

        prev_norm = current_norm
        final_result = result
    return final_result


def _solve_stokes(case_spec, N, nu_val, f_vals, x_range, y_range, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    degree_u = 2
    degree_p = 1
    cell_name = domain.topology.cell_name()
    P2 = element("Lagrange", cell_name, degree_u, shape=(2,))
    P1_el = element("Lagrange", cell_name, degree_p)
    TH = mixed_element([P2, P1_el])
    W = fem.functionspace(domain, TH)
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(domain, ScalarType((f_vals[0], f_vals[1])))
    nu = fem.Constant(domain, ScalarType(nu_val))

    a_form = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              - p * ufl.div(v) * ufl.dx
              + q * ufl.div(u) * ufl.dx)
    L_form = ufl.inner(f, v) * ufl.dx

    # Parse boundary conditions
    pde = case_spec.get("pde", {})
    bc_list_spec = pde.get("boundary_conditions", [])
    bcs = []
    has_outflow = False

    if isinstance(bc_list_spec, list) and len(bc_list_spec) > 0:
        for bc_item in bc_list_spec:
            bc_type = bc_item.get("type", "dirichlet").lower()
            location = bc_item.get("location", "")
            if bc_type in ["neumann", "outflow"] or "outflow" in location.lower():
                has_outflow = True
                continue
            if bc_type == "dirichlet":
                value = bc_item.get("value", [0.0, 0.0])
                if isinstance(value, (list, tuple)):
                    val = np.array([float(v_) for v_ in value])
                else:
                    val = np.array([float(value), 0.0])
                marker_func = _get_boundary_marker(location, x_range, y_range)
                if marker_func is not None:
                    facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
                    dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                    u_bc = fem.Function(V_sub)
                    u_bc.interpolate(lambda x, v=val: np.tile(v.reshape(2, 1), (1, x.shape[1])))
                    bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    elif isinstance(bc_list_spec, dict) and len(bc_list_spec) > 0:
        for loc, bc_info in bc_list_spec.items():
            if isinstance(bc_info, dict):
                bc_type = bc_info.get("type", "dirichlet").lower()
                value = bc_info.get("value", [0.0, 0.0])
            else:
                bc_type = "dirichlet"
                value = bc_info
            if bc_type in ["neumann", "outflow"]:
                has_outflow = True
                continue
            if isinstance(value, (list, tuple)):
                val = np.array([float(v_) for v_ in value])
            else:
                val = np.array([float(value), 0.0])
            marker_func = _get_boundary_marker(loc, x_range, y_range)
            if marker_func is not None:
                facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
                dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                u_bc = fem.Function(V_sub)
                u_bc.interpolate(lambda x, v=val: np.tile(v.reshape(2, 1), (1, x.shape[1])))
                bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # Default BCs if none parsed: no-slip on left/bottom/top, outflow on right
    if len(bcs) == 0:
        for marker_name in ["left", "bottom", "top"]:
            marker_func = _get_boundary_marker(marker_name, x_range, y_range)
            facets = mesh.locate_entities_boundary(domain, fdim, marker_func)
            dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
            u_bc = fem.Function(V_sub)
            u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
            bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
        has_outflow = True

    # Pin pressure if no outflow (all Dirichlet velocity)
    if not has_outflow:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q_sub),
            lambda x: np.isclose(x[0], x_range[0]) & np.isclose(x[1], y_range[0])
        )
        if len(p_dofs[0]) > 0:
            p_bc_val = fem.Function(Q_sub)
            p_bc_val.x.array[:] = 0.0
            bcs.append(fem.dirichletbc(p_bc_val, p_dofs, W.sub(1)))

    # Solve with MUMPS (handles indefinite saddle-point systems)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()

    # Extract velocity
    uh = wh.sub(0).collapse()

    # Evaluate on output grid
    x_pts = np.linspace(x_range[0], x_range[1], nx_out)
    y_pts = np.linspace(y_range[0], y_range[1], ny_out)
    XX, YY = np.meshgrid(x_pts, y_pts, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.flatten()
    points_3d[:, 1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    vel_mag = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        vals = vals.reshape(-1, 2)
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]

    u_grid = vel_mag.reshape((nx_out, ny_out))

    # Handle NaN at boundaries
    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.flatten())
        if np.sum(valid) > 0:
            coords_valid = np.column_stack([XX.flatten()[valid], YY.flatten()[valid]])
            vals_valid = u_grid.flatten()[valid]
            interp_nn = NearestNDInterpolator(coords_valid, vals_valid)
            nan_mask = np.isnan(u_grid.flatten())
            coords_nan = np.column_stack([XX.flatten()[nan_mask], YY.flatten()[nan_mask]])
            u_grid.flat[nan_mask] = interp_nn(coords_nan)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }
    return {"u": u_grid, "solver_info": solver_info}


def _get_boundary_marker(location, x_range, y_range):
    loc = location.lower().strip()
    if loc in ["left", "x_min", "x=0", "x0"]:
        return lambda x: np.isclose(x[0], x_range[0])
    elif loc in ["right", "x_max", "x=1", "x1"]:
        return lambda x: np.isclose(x[0], x_range[1])
    elif loc in ["bottom", "y_min", "y=0", "y0"]:
        return lambda x: np.isclose(x[1], y_range[0])
    elif loc in ["top", "y_max", "y=1", "y1"]:
        return lambda x: np.isclose(x[1], y_range[1])
    elif loc in ["all", "entire", "boundary"]:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    else:
        return None
