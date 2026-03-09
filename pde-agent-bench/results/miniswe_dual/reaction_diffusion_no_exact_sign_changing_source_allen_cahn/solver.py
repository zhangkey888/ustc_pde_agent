import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def _parse_ufl_expr(expr_str, x):
    """Parse a math expression string to UFL expression using spatial coordinates x."""
    ctx = {
        'x': x[0], 'y': x[1],
        'pi': ufl.pi,
        'sin': ufl.sin, 'cos': ufl.cos, 'exp': ufl.exp,
        'sqrt': ufl.sqrt, 'tanh': ufl.tanh,
        'log': ufl.ln,
    }
    safe_expr = expr_str.strip()
    safe_expr = safe_expr.replace('np.pi', 'pi')
    safe_expr = safe_expr.replace('numpy.pi', 'pi')
    safe_expr = safe_expr.replace('math.pi', 'pi')
    
    try:
        result = eval(safe_expr, {"__builtins__": {}}, ctx)
        return result
    except Exception as e:
        print(f"Warning: Could not parse expression '{expr_str}': {e}")
        return None


def _parse_numpy_expr(expr_str):
    """Parse a math expression string to a numpy callable for interpolation."""
    def func(X):
        x = X[0]
        y = X[1]
        ctx = {
            '__builtins__': {},
            'np': np, 'pi': np.pi,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'sqrt': np.sqrt, 'tanh': np.tanh, 'abs': np.abs,
            'log': np.log,
            'x': x, 'y': y,
        }
        safe_expr = expr_str.strip()
        safe_expr = safe_expr.replace('np.pi', 'pi')
        safe_expr = safe_expr.replace('numpy.pi', 'pi')
        safe_expr = safe_expr.replace('math.pi', 'pi')
        result = eval(safe_expr, ctx)
        return np.asarray(result, dtype=np.float64)
    return func


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion (Allen-Cahn) equation."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec - handle both direct and oracle_config structures
    # The actual case_spec may have oracle_config wrapping the pde info
    oracle_config = case_spec.get("oracle_config", {})
    pde_spec = oracle_config.get("pde", case_spec.get("pde", {}))
    
    # Get pde_params (nested structure)
    pde_params = pde_spec.get("pde_params", {})
    
    # Get epsilon (diffusion coefficient) - check multiple locations
    epsilon = pde_params.get("epsilon", None)
    if epsilon is None:
        epsilon = pde_spec.get("epsilon", None)
    if epsilon is None:
        epsilon = case_spec.get("epsilon", None)
    if epsilon is None:
        epsilon = 0.05  # Default for this problem
    epsilon = float(epsilon)
    
    # Get reaction parameters
    reaction_info = pde_params.get("reaction", {})
    reaction_type = reaction_info.get("type", pde_spec.get("reaction_type", "allen_cahn"))
    reaction_lambda = reaction_info.get("lambda", None)
    if reaction_lambda is not None:
        reaction_lambda = float(reaction_lambda)
    
    # Also check older format
    if reaction_lambda is None:
        reaction_params = pde_spec.get("reaction_params", {})
        coeff = reaction_params.get("coefficient", None)
        if coeff is not None:
            reaction_lambda = float(coeff)
    
    # Get source term string
    source_expr_str = pde_spec.get("source_term", "3*cos(3*pi*x)*sin(2*pi*y)")
    
    # Time parameters - with hardcoded defaults for this problem
    time_params = pde_spec.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_val = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    if t_end is None:
        t_end = 0.2
    if dt_val is None:
        dt_val = 0.005
    t_end = float(t_end)
    dt_val = float(dt_val)
    
    # Force transient
    is_transient = True
    
    # Initial condition
    ic_expr_str = pde_spec.get("initial_condition", "0.2*sin(3*pi*x)*sin(2*pi*y)")
    
    # Boundary conditions from case_spec
    bc_config = oracle_config.get("bc", case_spec.get("bc", {}))
    dirichlet_config = bc_config.get("dirichlet", {})
    bc_value_str = dirichlet_config.get("value", "0.0")
    
    # Mesh resolution and element degree
    mesh_config = oracle_config.get("mesh", {})
    fem_config = oracle_config.get("fem", {})
    
    mesh_resolution = case_spec.get("mesh_resolution", None)
    element_degree = case_spec.get("element_degree", None)
    
    if mesh_resolution is None:
        mesh_resolution = mesh_config.get("resolution", None)
    if element_degree is None:
        element_degree = fem_config.get("degree", None)
    
    # Choose good defaults - aim for accuracy
    if mesh_resolution is None:
        # For epsilon=0.05, interface width ~ sqrt(epsilon) ~ 0.22
        # Need mesh to resolve this
        if epsilon >= 0.05:
            mesh_resolution = 140
        elif epsilon >= 0.01:
            mesh_resolution = 200
        else:
            mesh_resolution = 256
    
    if element_degree is None:
        element_degree = 1
    
    N = int(mesh_resolution)
    deg = int(element_degree)
    
    # Build mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    # Spatial coordinates
    x_coord = ufl.SpatialCoordinate(domain)
    
    # Build source term as UFL expression
    f_ufl = _parse_ufl_expr(source_expr_str, x_coord)
    if f_ufl is None:
        f_ufl = 3.0 * ufl.cos(3.0 * ufl.pi * x_coord[0]) * ufl.sin(2.0 * ufl.pi * x_coord[1])
    
    # Build initial condition callable
    ic_func = _parse_numpy_expr(ic_expr_str)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Parse BC value
    try:
        bc_val = float(bc_value_str)
        bc = fem.dirichletbc(PETSc.ScalarType(bc_val), dofs, V)
    except (ValueError, TypeError):
        bc_fn = fem.Function(V)
        bc_fn.interpolate(_parse_numpy_expr(str(bc_value_str)))
        bc = fem.dirichletbc(bc_fn, dofs)
    
    bcs = [bc]
    
    # Functions
    u = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    # Set initial condition
    u_n.interpolate(ic_func)
    u.x.array[:] = u_n.x.array[:]
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(ic_func)
    
    # Constants
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # Build reaction term R(u)
    # For Allen-Cahn: ∂u/∂t - ε∇²u + R(u) = f
    # R(u) = λ(u³ - u) where λ is the reaction coefficient
    if reaction_type in ("allen_cahn", "allen-cahn"):
        if reaction_lambda is not None:
            lam = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))
            R_u = lam * (u**3 - u)
        else:
            # Default: R(u) = (1/ε)(u³ - u)
            R_u = (1.0 / eps_c) * (u**3 - u)
    elif reaction_type == "cubic":
        R_u = u**3
    elif reaction_type == "linear":
        coeff = reaction_lambda if reaction_lambda is not None else 1.0
        R_u = fem.Constant(domain, PETSc.ScalarType(coeff)) * u
    else:
        if reaction_lambda is not None:
            lam = fem.Constant(domain, PETSc.ScalarType(reaction_lambda))
            R_u = lam * (u**3 - u)
        else:
            R_u = (1.0 / eps_c) * (u**3 - u)
    
    # Backward Euler weak form:
    # (u - u_n)/dt * v * dx + ε * grad(u) · grad(v) * dx + R(u) * v * dx - f * v * dx = 0
    F_form = (
        (u - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + R_u * v * ufl.dx
        - f_ufl * v * ufl.dx
    )
    
    # Jacobian
    J_form = ufl.derivative(F_form, u)
    
    # Setup nonlinear solver
    ksp_type_used = "gmres"
    pc_type_used = "ilu"
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "ksp_rtol": 1e-10,
        "ksp_max_it": 2000,
    }
    
    problem = petsc.NonlinearProblem(
        F_form, u,
        bcs=bcs,
        J=J_form,
        petsc_options_prefix="nlsolve_",
        petsc_options=petsc_options,
    )
    
    snes = problem.solver
    
    # Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt_val))
    nonlinear_iterations = []
    total_linear_iters = 0
    
    for step in range(n_steps):
        t += dt_val
        
        # Use previous solution as initial guess
        u.x.array[:] = u_n.x.array[:]
        
        # Solve
        problem.solve()
        
        # Get iteration info
        newton_its = snes.getIterationNumber()
        nonlinear_iterations.append(newton_its)
        
        lin_its = snes.getLinearSolveIterations()
        total_linear_iters += lin_its
        
        reason = snes.getConvergedReason()
        if reason < 0:
            print(f"WARNING: SNES did not converge at step {step}, t={t:.4f}, reason={reason}")
        
        # Update previous solution
        u_n.x.array[:] = u.x.array[:]
    
    # Evaluate on 70x70 grid
    nx_out, ny_out = 70, 70
    
    # Check if output grid is specified in case_spec
    output_config = oracle_config.get("output", {})
    grid_config = output_config.get("grid", {})
    if grid_config:
        nx_out = grid_config.get("nx", 70)
        ny_out = grid_config.get("ny", 70)
        bbox = grid_config.get("bbox", [0, 1, 0, 1])
        x_min, x_max, y_min, y_max = bbox
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    
    xs = np.linspace(x_min, x_max, nx_out)
    ys = np.linspace(y_min, y_max, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    def eval_function_on_grid(func, points_3d, colliding_cells):
        values = np.full(points_3d.shape[0], np.nan)
        pts_list = []
        cells_list = []
        idx_list = []
        
        for i in range(points_3d.shape[0]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                pts_list.append(points_3d[i])
                cells_list.append(links[0])
                idx_list.append(i)
        
        if len(pts_list) > 0:
            pts_arr = np.array(pts_list)
            cells_arr = np.array(cells_list, dtype=np.int32)
            vals = func.eval(pts_arr, cells_arr)
            values[idx_list] = vals.flatten()
        
        return values
    
    u_values = eval_function_on_grid(u, points_3d, colliding_cells)
    u_grid = u_values.reshape((nx_out, ny_out))
    
    u_init_values = eval_function_on_grid(u_initial_func, points_3d, colliding_cells)
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Handle any NaN values (boundary points that might not be found)
    if np.any(np.isnan(u_grid)):
        nan_count = np.isnan(u_grid).sum()
        print(f"Warning: {nan_count} NaN values in output, filling with nearest")
        from scipy.ndimage import generic_filter
        # Simple fill: replace NaN with 0 (boundary value)
        u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    if np.any(np.isnan(u_initial_grid)):
        u_initial_grid = np.nan_to_num(u_initial_grid, nan=0.0)
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": 1e-10,
            "iterations": total_linear_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import json
    
    # Load the actual case_spec if available
    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "reaction_diffusion",
                "pde_params": {
                    "epsilon": 0.05,
                    "reaction": {"type": "allen_cahn", "lambda": 2.0}
                },
                "source_term": "3*cos(3*pi*x)*sin(2*pi*y)",
                "initial_condition": "0.2*sin(3*pi*x)*sin(2*pi*y)",
                "time": {
                    "t0": 0.0,
                    "t_end": 0.2,
                    "dt": 0.005,
                    "scheme": "backward_euler",
                },
            },
            "bc": {"dirichlet": {"on": "all", "value": "0.0"}},
            "output": {
                "format": "npz",
                "field": "scalar",
                "grid": {"bbox": [0, 1, 0, 1], "nx": 70, "ny": 70}
            },
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Mesh: {result['solver_info']['mesh_resolution']}, Degree: {result['solver_info']['element_degree']}")
    print(f"Steps: {result['solver_info']['n_steps']}, dt: {result['solver_info']['dt']}")
    print(f"Newton iters per step: {result['solver_info']['nonlinear_iterations']}")
