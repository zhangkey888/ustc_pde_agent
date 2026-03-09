import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve the Helmholtz equation: -∇²u - k²u = f in Ω, u = g on ∂Ω
    """
    # Parse case_spec - handle both direct oracle_config and nested structure
    oc = case_spec.get("oracle_config", case_spec)
    pde = oc.get("pde", {})
    
    # Wavenumber k
    params = pde.get("pde_params", {})
    k_val = float(params.get("k", params.get("wave_number", pde.get("wavenumber", pde.get("k", 12.0)))))
    
    # Source term
    source_str = pde.get("source_term", pde.get("source", "0.0"))
    
    # Domain
    domain_info = oc.get("domain", {})
    domain_type = domain_info.get("type", "unit_square")
    
    # Output grid
    output_cfg = oc.get("output", {})
    grid_cfg = output_cfg.get("grid", {})
    bbox = grid_cfg.get("bbox", [0, 1, 0, 1])
    nx_out = grid_cfg.get("nx", 50)
    ny_out = grid_cfg.get("ny", 50)
    x_range = [bbox[0], bbox[1]]
    y_range = [bbox[2], bbox[3]]
    
    # Boundary conditions
    bc_cfg = oc.get("bc", {})
    dirichlet_cfg = bc_cfg.get("dirichlet", {})
    bc_value_str = dirichlet_cfg.get("value", "0.0")
    
    # Solver parameters
    element_degree = 2
    N = 64  # Good for k=12 with P2 elements
    
    comm = MPI.COMM_WORLD
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = _parse_source_ufl(source_str, x, domain)
    
    # Bilinear form: ∇u·∇v - k²uv
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k_val**2) * u * v * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc_func = _make_bc_interpolation_func(bc_value_str)
    u_bc.interpolate(bc_func)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Solve with direct solver (robust for indefinite Helmholtz)
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(u_sol, domain, x_range, y_range, nx_out, ny_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": iterations,
        },
    }


def _parse_source_ufl(source_str, x, domain):
    """Parse source term string to UFL expression or fem.Constant."""
    if source_str is None:
        return fem.Constant(domain, ScalarType(0.0))
    try:
        val = float(source_str)
        return fem.Constant(domain, ScalarType(val))
    except (ValueError, TypeError):
        # Try to parse as symbolic expression
        import sympy as sp
        sx, sy = sp.symbols("x y", real=True)
        expr_sympy = sp.sympify(str(source_str), locals={"x": sx, "y": sy, "pi": sp.pi})
        return _sympy_to_ufl(expr_sympy, x)


def _sympy_to_ufl(expr_sympy, x):
    """Convert a sympy expression to UFL using SpatialCoordinate."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    
    # Convert sympy to string and evaluate with UFL
    expr_str = str(expr_sympy)
    
    # Replace sympy functions with UFL equivalents
    pi = ufl.pi
    x0 = x[0]
    x1 = x[1]
    
    # Use sympy lambdify approach
    from sympy import sin, cos, exp, sqrt, pi as sp_pi
    
    # Substitute and convert
    expr_float = expr_sympy.subs(sp_pi, float(np.pi))
    
    # Build UFL expression by traversal
    return _sympy_expr_to_ufl(expr_sympy, x)


def _sympy_expr_to_ufl(expr, x):
    """Recursively convert sympy expression to UFL."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    
    if expr == sx:
        return x[0]
    elif expr == sy:
        return x[1]
    elif isinstance(expr, sp.Number):
        return float(expr)
    elif isinstance(expr, sp.Pi):
        return ufl.pi
    elif isinstance(expr, sp.Add):
        result = _sympy_expr_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result + _sympy_expr_to_ufl(arg, x)
        return result
    elif isinstance(expr, sp.Mul):
        result = _sympy_expr_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result * _sympy_expr_to_ufl(arg, x)
        return result
    elif isinstance(expr, sp.Pow):
        base = _sympy_expr_to_ufl(expr.args[0], x)
        exp_val = _sympy_expr_to_ufl(expr.args[1], x)
        return base ** exp_val
    elif isinstance(expr, sp.sin):
        return ufl.sin(_sympy_expr_to_ufl(expr.args[0], x))
    elif isinstance(expr, sp.cos):
        return ufl.cos(_sympy_expr_to_ufl(expr.args[0], x))
    elif isinstance(expr, sp.exp):
        return ufl.exp(_sympy_expr_to_ufl(expr.args[0], x))
    elif isinstance(expr, sp.Rational):
        return float(expr)
    elif isinstance(expr, sp.Integer):
        return float(expr)
    elif isinstance(expr, sp.Float):
        return float(expr)
    else:
        # Fallback: try to convert to float
        try:
            return float(expr)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert sympy expression to UFL: {expr} (type: {type(expr)})")


def _make_bc_interpolation_func(bc_value_str):
    """Create a numpy interpolation function from BC string expression."""
    if bc_value_str is None:
        return lambda x: np.zeros(x.shape[1])
    
    try:
        val = float(bc_value_str)
        return lambda x: np.full(x.shape[1], val)
    except (ValueError, TypeError):
        pass
    
    # Parse string expression for numpy evaluation
    def bc_func(x):
        x0 = x[0]
        x1 = x[1]
        pi = np.pi
        
        # Build safe evaluation context
        safe_dict = {
            "x": x0, "y": x1,
            "sin": np.sin, "cos": np.cos,
            "exp": np.exp, "sqrt": np.sqrt,
            "pi": np.pi, "np": np,
            "abs": np.abs,
        }
        
        expr = str(bc_value_str)
        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            if isinstance(result, (int, float)):
                return np.full(x0.shape, float(result))
            return np.asarray(result, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to evaluate BC expression '{expr}': {e}")
    
    return bc_func


def _evaluate_on_grid(u_func, domain, x_range, y_range, nx, ny):
    """Evaluate a FEM function on a regular grid."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()
    
    u_values = np.nan_to_num(u_values, nan=0.0)
    return u_values.reshape((nx, ny))
