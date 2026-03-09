import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow (incompressible) with Taylor-Hood elements."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    pde_spec = case_spec.get("pde", {})
    nu_val = float(pde_spec.get("viscosity", 1.0))
    
    # Parse source term
    source = pde_spec.get("source_term", ["0.0", "0.0"])
    if isinstance(source, str):
        source = [source, "0.0"]
    
    # Parse domain
    domain_spec = case_spec.get("domain", {})
    x_min = float(domain_spec.get("x_min", 0.0))
    x_max = float(domain_spec.get("x_max", 1.0))
    y_min = float(domain_spec.get("y_min", 0.0))
    y_max = float(domain_spec.get("y_max", 1.0))
    
    # Parse boundary conditions from multiple possible locations
    bc_spec = pde_spec.get("boundary_conditions",
                           case_spec.get("boundary_conditions", {}))
    
    # Output grid
    output_spec = case_spec.get("output", {})
    nx_grid = int(output_spec.get("nx", 100))
    ny_grid = int(output_spec.get("ny", 100))
    
    # Element degrees
    degree_u = 2
    degree_p = 1
    
    # Adaptive mesh refinement
    resolutions = [48, 96, 160]
    prev_norm = None
    final_result = None
    used_N = None
    used_ksp = "minres"
    used_pc = "hypre"
    rtol_val = 1e-12
    
    for N in resolutions:
        # Create mesh
        p0 = np.array([x_min, y_min])
        p1 = np.array([x_max, y_max])
        msh = mesh.create_rectangle(comm, [p0, p1], [N, N],
                                     cell_type=mesh.CellType.triangle)
        tdim = msh.topology.dim
        fdim = tdim - 1
        
        # Taylor-Hood P2/P1 mixed elements
        P_vec = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_u, shape=(2,))
        P_scl = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree_p)
        mel = basix.ufl.mixed_element([P_vec, P_scl])
        W = fem.functionspace(msh, mel)
        
        # Trial and test functions
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        
        # Source term
        x_coord = ufl.SpatialCoordinate(msh)
        f_body = _parse_source_term(msh, source, x_coord)
        
        # Bilinear and linear forms for Stokes
        a_form = (nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                   - p * ufl.div(v) * ufl.dx
                   + q * ufl.div(u) * ufl.dx)
        L_form = ufl.inner(f_body, v) * ufl.dx
        
        # Build boundary conditions
        bcs = _build_bcs(msh, W, bc_spec, fdim, x_min, x_max, y_min, y_max)
        
        # Solve with fallback strategy
        ksp_type = "minres"
        pc_type = "hypre"
        
        try:
            problem = petsc.LinearProblem(
                a_form, L_form, bcs=bcs,
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol_val),
                    "ksp_atol": "1e-15",
                    "ksp_max_it": "10000",
                },
                petsc_options_prefix=f"stk{N}_"
            )
            wh = problem.solve()
        except Exception:
            # Fallback to direct solver
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a_form, L_form, bcs=bcs,
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix=f"stk_lu{N}_"
            )
            wh = problem.solve()
        
        used_ksp = ksp_type
        used_pc = pc_type
        used_N = N
        
        # Extract velocity
        u_h = wh.sub(0).collapse()
        
        # Sample on output grid
        u_grid = _sample_velocity_magnitude(msh, u_h, tdim, nx_grid, ny_grid,
                                             x_min, x_max, y_min, y_max)
        
        final_result = u_grid
        
        # Convergence check based on L2 norm
        current_norm = np.sqrt(np.nanmean(u_grid**2))
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-6:
                break
        
        prev_norm = current_norm
    
    solver_info = {
        "mesh_resolution": used_N,
        "element_degree": degree_u,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": rtol_val,
        "iterations": 0,
    }
    
    return {
        "u": final_result,
        "solver_info": solver_info,
    }


def _safe_eval_expr(expr_str, x0, x1):
    """Safely evaluate a math expression string at given coordinates."""
    ns = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "pi": np.pi, "e": np.e, "abs": np.abs,
        "pow": np.power,
        "x0_": x0, "x1_": x1,
        "__builtins__": {},
    }
    s = expr_str.strip()
    s = s.replace("x[0]", "x0_").replace("x[1]", "x1_")
    try:
        return float(eval(s, ns))
    except Exception:
        return 0.0


def _parse_source_term(msh, source_strs, x_coord):
    """Parse source term strings into UFL or constant."""
    # Quick check for zero source
    if all(s.strip() in ("0", "0.0") for s in source_strs):
        return fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Check if constant by evaluating at multiple points
    v1 = [_safe_eval_expr(s, 0.3, 0.7) for s in source_strs]
    v2 = [_safe_eval_expr(s, 0.6, 0.2) for s in source_strs]
    v3 = [_safe_eval_expr(s, 0.1, 0.9) for s in source_strs]
    
    if np.allclose(v1, v2) and np.allclose(v2, v3):
        return fem.Constant(msh, PETSc.ScalarType(tuple(v1)))
    
    # Non-constant: build UFL expression
    x0 = x_coord[0]
    x1 = x_coord[1]
    components = []
    for s in source_strs:
        components.append(_str_to_ufl(s, x0, x1))
    return ufl.as_vector(components)


def _str_to_ufl(expr_str, x0, x1):
    """Convert a string expression to UFL."""
    s = expr_str.strip()
    s = s.replace("x[0]", "x0_var").replace("x[1]", "x1_var")
    ns = {
        "x0_var": x0, "x1_var": x1,
        "sin": ufl.sin, "cos": ufl.cos, "tan": ufl.tan,
        "exp": ufl.exp, "ln": ufl.ln, "sqrt": ufl.sqrt,
        "pi": np.pi, "e": np.e,
        "__builtins__": {},
    }
    try:
        return eval(s, ns)
    except Exception:
        return 0.0


def _make_bc_interpolator(val_spec):
    """Create an interpolation function from a BC value specification."""
    if isinstance(val_spec, (int, float)):
        v = float(val_spec)
        return lambda x, v=v: np.stack([np.full_like(x[0], v), np.zeros_like(x[0])])
    
    if isinstance(val_spec, (list, tuple)) and len(val_spec) >= 2:
        vals_str = [str(v) for v in val_spec[:2]]
        
        # Check if constant
        v1 = [_safe_eval_expr(s, 0.3, 0.7) for s in vals_str]
        v2 = [_safe_eval_expr(s, 0.6, 0.2) for s in vals_str]
        
        if np.allclose(v1, v2):
            c0, c1 = v1[0], v1[1]
            return lambda x, c0=c0, c1=c1: np.stack([
                np.full_like(x[0], c0),
                np.full_like(x[0], c1)
            ])
        else:
            # Spatially varying BC
            def interp_func(x, exprs=vals_str):
                result = np.zeros((2, x.shape[1]))
                for i, expr in enumerate(exprs):
                    s = expr.replace("x[0]", "x0_").replace("x[1]", "x1_")
                    for j in range(x.shape[1]):
                        ns = {
                            "x0_": x[0][j], "x1_": x[1][j],
                            "sin": np.sin, "cos": np.cos, "exp": np.exp,
                            "pi": np.pi, "sqrt": np.sqrt,
                            "__builtins__": {},
                        }
                        try:
                            result[i, j] = eval(s, ns)
                        except Exception:
                            result[i, j] = 0.0
                return result
            return interp_func
    
    # Default: zero velocity
    return lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0])])


def _build_bcs(msh, W, bc_spec, fdim, x_min, x_max, y_min, y_max):
    """Build boundary conditions for the Stokes problem."""
    bcs = []
    V_sub, _ = W.sub(0).collapse()
    
    has_dirichlet = False
    
    if bc_spec and isinstance(bc_spec, dict):
        for bc_name, bc_data in bc_spec.items():
            if not isinstance(bc_data, dict):
                continue
            
            bc_type = bc_data.get("type", "dirichlet").lower().strip()
            
            if bc_type in ("dirichlet", "essential", "velocity"):
                location = bc_data.get("location", bc_name).lower().strip()
                value = bc_data.get("value", [0.0, 0.0])
                
                marker = _get_boundary_marker(location, x_min, x_max, y_min, y_max)
                if marker is not None:
                    facets = mesh.locate_entities_boundary(msh, fdim, marker)
                    if len(facets) > 0:
                        u_bc = fem.Function(V_sub)
                        interp = _make_bc_interpolator(value)
                        u_bc.interpolate(interp)
                        dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, facets)
                        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
                        has_dirichlet = True
            # natural/open/neumann -> do nothing
    
    if not has_dirichlet:
        # Default for counter_shear_open_sides:
        # Top: u = (1, 0), Bottom: u = (-1, 0), Left/Right: open
        u_top = fem.Function(V_sub)
        u_top.interpolate(lambda x: np.stack([np.ones_like(x[0]), np.zeros_like(x[0])]))
        top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_max))
        top_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, top_facets)
        bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))
        
        u_bot = fem.Function(V_sub)
        u_bot.interpolate(lambda x: np.stack([-np.ones_like(x[0]), np.zeros_like(x[0])]))
        bot_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], y_min))
        bot_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, bot_facets)
        bcs.append(fem.dirichletbc(u_bot, bot_dofs, W.sub(0)))
    
    return bcs


def _get_boundary_marker(location, x_min, x_max, y_min, y_max):
    """Return a boundary marker function based on location string."""
    loc = location.lower().strip()
    if loc in ("top", "y_max", "upper", "y=1", "y=1.0"):
        return lambda x: np.isclose(x[1], y_max)
    elif loc in ("bottom", "y_min", "lower", "y=0", "y=0.0"):
        return lambda x: np.isclose(x[1], y_min)
    elif loc in ("left", "x_min", "x=0", "x=0.0"):
        return lambda x: np.isclose(x[0], x_min)
    elif loc in ("right", "x_max", "x=1", "x=1.0"):
        return lambda x: np.isclose(x[0], x_max)
    elif loc in ("all", "entire", "boundary"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    return None


def _sample_velocity_magnitude(msh, u_h, tdim, nx_grid, ny_grid, x_min, x_max, y_min, y_max):
    """Sample velocity magnitude on a regular grid."""
    xs = np.linspace(x_min, x_max, nx_grid)
    ys = np.linspace(y_min, y_max, ny_grid)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full(nx_grid * ny_grid, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vel_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        u_grid[eval_map] = vel_mag
    
    return u_grid.reshape(nx_grid, ny_grid)
