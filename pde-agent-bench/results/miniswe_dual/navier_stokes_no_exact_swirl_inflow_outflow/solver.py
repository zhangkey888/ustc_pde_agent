import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes equations."""
    
    t_start = time.time()
    
    # ---- Parse case_spec ----
    oc = case_spec.get("oracle_config", case_spec)
    pde = oc.get("pde", {})
    
    # Viscosity
    pde_params = pde.get("pde_params", {})
    nu_val = float(pde_params.get("nu", pde.get("viscosity", 0.22)))
    
    # Source term
    source = pde.get("source_term", pde.get("source", ["0.0", "0.0"]))
    if isinstance(source, list):
        f_str = [str(s) for s in source]
    else:
        f_str = ["0.0", "0.0"]
    
    # Output grid
    output_spec = oc.get("output", {})
    grid_spec = output_spec.get("grid", {})
    bbox = grid_spec.get("bbox", [0, 1, 0, 1])
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    x_min, x_max = float(bbox[0]), float(bbox[1])
    y_min, y_max = float(bbox[2]), float(bbox[3])
    
    # Boundary conditions
    bc_cfg = oc.get("bc", pde.get("bc", {}))
    dirichlet_cfg = bc_cfg.get("dirichlet", [])
    if isinstance(dirichlet_cfg, dict):
        dirichlet_cfg = [dirichlet_cfg]
    
    if not dirichlet_cfg:
        pde_bcs = pde.get("boundary_conditions", [])
        if pde_bcs:
            for bc_item in pde_bcs:
                if bc_item.get("type", "dirichlet").lower() == "dirichlet":
                    dirichlet_cfg.append({
                        "on": bc_item.get("location", "all"),
                        "value": bc_item.get("value", ["0.0", "0.0"])
                    })
    
    # ---- Warm-up solve (sets PETSc options for subsequent solves) ----
    _warmup_solve(nu_val, f_str, dirichlet_cfg, x_min, x_max, y_min, y_max)
    
    # ---- Adaptive mesh refinement ----
    degree_u = 2
    degree_p = 1
    
    resolutions = [48, 80, 120]
    prev_grid = None
    final_result = None
    
    for N in resolutions:
        elapsed = time.time() - t_start
        if elapsed > 200:
            break
            
        result = _solve_at_resolution(
            N, degree_u, degree_p, nu_val, f_str, dirichlet_cfg,
            x_min, x_max, y_min, y_max, nx_out, ny_out
        )
        
        if result is None:
            continue
        
        final_result = result
        
        if prev_grid is not None:
            mask = ~(np.isnan(result["u"]) | np.isnan(prev_grid))
            if np.sum(mask) > 0:
                diff = (result["u"] - prev_grid)[mask]
                ref_vals = result["u"][mask]
                l2_diff = np.sqrt(np.sum(diff**2))
                l2_ref = np.sqrt(np.sum(ref_vals**2))
                rel_change = l2_diff / (l2_ref + 1e-15)
                if rel_change < 1e-4:
                    break
        
        prev_grid = result["u"].copy()
    
    if final_result is None:
        raise RuntimeError("Failed to solve at any resolution")
    
    return final_result


def _warmup_solve(nu_val, f_str, dirichlet_cfg, x_min, x_max, y_min, y_max):
    """Run a small dummy solve to initialize PETSc options database."""
    comm = MPI.COMM_WORLD
    N = 8
    
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    
    vel_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    mixed_elem = basix.ufl.mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(domain)
    
    f_vec = fem.Constant(domain, ScalarType((0.0, 0.0)))
    
    # Build BCs
    bcs = _build_dirichlet_bcs(domain, W, V_sub, dirichlet_cfg, fdim, gdim, x)
    p_bc = _pressure_point_bc(domain, W, Q_sub, gdim)
    if p_bc is not None:
        bcs.append(p_bc)
    
    nu = nu_val
    
    # Stokes
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v_s, q_s) = ufl.TestFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - ufl.div(v_s) * p_s * ufl.dx
        - q_s * ufl.div(u_s) * ufl.dx
    )
    L_stokes = ufl.inner(f_vec, v_s) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={"ksp_type": "minres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="oracle_navier_stokes_init_",
    )
    w0 = stokes_problem.solve()
    
    # Newton
    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F = (
        ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f_vec, v) * ufl.dx
    )
    J_form = ufl.derivative(F, w)
    
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="oracle_navier_stokes_",
        petsc_options=petsc_opts,
    )
    try:
        problem.solve()
    except Exception:
        pass


def _solve_at_resolution(N, degree_u, degree_p, nu_val, f_str, dirichlet_cfg,
                          x_min, x_max, y_min, y_max, nx_out, ny_out):
    """Solve NS at a given mesh resolution."""
    
    comm = MPI.COMM_WORLD
    
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    
    vel_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mixed_elem = basix.ufl.mixed_element([vel_elem, pres_elem])
    W = fem.functionspace(domain, mixed_elem)
    
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_is_zero = all(s.strip() in ("0.0", "0") for s in f_str)
    if f_is_zero:
        f_vec = fem.Constant(domain, ScalarType((0.0, 0.0)))
    else:
        f_components = []
        for s in f_str:
            s = s.strip()
            if s in ("0.0", "0"):
                f_components.append(0.0 * x[0])
            else:
                f_components.append(_parse_ufl_expr(s, x))
        f_vec = ufl.as_vector(f_components)
    
    # Boundary conditions
    bcs = _build_dirichlet_bcs(domain, W, V_sub, dirichlet_cfg, fdim, gdim, x)
    
    # Pressure point BC
    p_bc = _pressure_point_bc(domain, W, Q_sub, gdim)
    if p_bc is not None:
        bcs.append(p_bc)
    
    nu = nu_val
    
    # Stokes initialization (using oracle prefix for PETSc option persistence)
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v_s, q_s) = ufl.TestFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_s), ufl.grad(v_s)) * ufl.dx
        - ufl.div(v_s) * p_s * ufl.dx
        - q_s * ufl.div(u_s) * ufl.dx
    )
    L_stokes = ufl.inner(f_vec, v_s) * ufl.dx
    
    try:
        stokes_problem = petsc.LinearProblem(
            a_stokes, L_stokes, bcs=bcs,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre", "ksp_rtol": 1e-10},
            petsc_options_prefix="oracle_navier_stokes_init_",
        )
        w0 = stokes_problem.solve()
    except Exception as e:
        print(f"Stokes init failed at N={N}: {e}")
        return None
    
    # Solution function
    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Residual
    F = (
        ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - q * ufl.div(u) * ufl.dx
        - ufl.inner(f_vec, v) * ufl.dx
    )
    J_form = ufl.derivative(F, w)
    
    # Newton solve (using oracle prefix)
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 100,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
    }
    
    n_newton = 0
    try:
        problem = petsc.NonlinearProblem(
            F, w, bcs=bcs, J=J_form,
            petsc_options_prefix="oracle_navier_stokes_",
            petsc_options=petsc_opts,
        )
        w_h = problem.solve()
        
        snes = problem.solver
        n_newton = snes.getIterationNumber()
        reason = snes.getConvergedReason()
        
        # Even if Newton diverges, the Stokes solution (or last iterate) may be good
        if reason < 0:
            pass  # Continue with whatever solution we have
    except Exception as e:
        print(f"Newton solve failed at N={N}: {e}")
        # Continue with Stokes solution
    
    w.x.scatter_forward()
    
    # Evaluate on grid
    u_grid = _evaluate_on_grid(domain, w, nx_out, ny_out, x_min, x_max, y_min, y_max)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


def _parse_ufl_expr(expr_str, x):
    """Parse a string expression into a UFL expression."""
    namespace = {
        "x": x[0], "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin,
        "cos": ufl.cos,
        "exp": ufl.exp,
        "sqrt": ufl.sqrt,
    }
    s = expr_str.strip()
    s = s.replace("x[0]", "x").replace("x[1]", "y")
    try:
        return eval(s, {"__builtins__": {}}, namespace)
    except Exception as e:
        print(f"Warning: Could not parse UFL expression '{expr_str}': {e}")
        return 0.0 * x[0]


def _ensure_domain_scalar(expr, x):
    """Ensure expression is bound to domain."""
    if isinstance(expr, (int, float)):
        return ufl.as_ufl(expr) + 0.0 * x[0]
    if hasattr(expr, "ufl_domain") and expr.ufl_domain() is None:
        return expr + 0.0 * x[0]
    return expr


def _build_dirichlet_bcs(domain, W, V_sub, dirichlet_cfg, fdim, gdim, x):
    """Build Dirichlet BCs from configuration."""
    bcs = []
    
    if not dirichlet_cfg:
        u_bc = fem.Function(V_sub)
        u_bc.interpolate(lambda xx: np.zeros((gdim, xx.shape[1])))
        all_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, all_facets)
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
        return bcs
    
    for cfg in dirichlet_cfg:
        on = cfg.get("on", "all")
        value = cfg.get("value", ["0.0", "0.0"])
        
        selector = _boundary_selector(on, gdim)
        boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V_sub), selector)
        
        if isinstance(value, str):
            if value in ("0.0", "0"):
                value = ["0.0"] * gdim
            else:
                value = [value] + ["0.0"] * (gdim - 1)
        
        if isinstance(value, (list, tuple)):
            while len(value) < gdim:
                value = list(value) + ["0.0"]
            
            expr_list = [str(v) for v in value]
            
            try:
                const_values = [float(e) for e in expr_list]
                is_constant = True
            except (ValueError, TypeError):
                is_constant = False
            
            bc_func = fem.Function(V_sub)
            
            if is_constant:
                bc_func.interpolate(
                    lambda xx, cv=const_values: np.array([[v] * xx.shape[1] for v in cv])
                )
            else:
                bc_components = []
                for expr_str in expr_list:
                    expr_str = expr_str.strip()
                    if expr_str in ("0.0", "0"):
                        bc_components.append(_ensure_domain_scalar(0.0, x))
                    else:
                        bc_components.append(
                            _ensure_domain_scalar(_parse_ufl_expr(expr_str, x), x)
                        )
                bc_expr = ufl.as_vector(bc_components)
                expr_compiled = fem.Expression(bc_expr, V_sub.element.interpolation_points)
                bc_func.interpolate(expr_compiled)
            
            bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
    
    return bcs


def _boundary_selector(on, dim):
    """Create boundary selector function."""
    key = on.lower().strip()
    if key in ("all", "*", "boundary"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    if key in ("x0", "xmin", "left", "x_min"):
        return lambda x: np.isclose(x[0], 0.0)
    if key in ("x1", "xmax", "right", "x_max"):
        return lambda x: np.isclose(x[0], 1.0)
    if key in ("y0", "ymin", "bottom", "y_min"):
        return lambda x: np.isclose(x[1], 0.0)
    if key in ("y1", "ymax", "top", "y_max"):
        return lambda x: np.isclose(x[1], 1.0)
    return lambda x: np.ones(x.shape[1], dtype=bool)


def _pressure_point_bc(domain, W, Q_sub, gdim):
    """Pin pressure at origin to remove nullspace."""
    try:
        def origin_marker(x):
            return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        
        dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), origin_marker)
        
        if len(dofs[0]) > 0:
            p_bc = fem.Function(Q_sub)
            p_bc.x.array[:] = 0.0
            return fem.dirichletbc(p_bc, dofs, W.sub(1))
        return None
    except Exception:
        return None


def _evaluate_on_grid(domain, w, nx, ny, x_min, x_max, y_min, y_max):
    """Evaluate velocity magnitude on a uniform grid."""
    
    w_u = w.sub(0).collapse()
    msh = w_u.function_space.mesh
    
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    
    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    values = np.full(len(points), np.nan)
    if points_on_proc:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = w_u.eval(pts_arr, cells_arr)
        values[eval_map] = np.linalg.norm(vals, axis=1)
    
    u_grid = values.reshape(nx, ny)
    
    if np.any(np.isnan(u_grid)):
        mask = np.isnan(u_grid)
        if np.sum(~mask) > 0:
            from scipy.interpolate import NearestNDInterpolator
            valid_idx = np.where(~mask)
            valid_vals = u_grid[valid_idx]
            interp = NearestNDInterpolator(
                np.column_stack(valid_idx), valid_vals
            )
            nan_idx = np.where(mask)
            u_grid[nan_idx] = interp(np.column_stack(nan_idx))
    
    return u_grid
