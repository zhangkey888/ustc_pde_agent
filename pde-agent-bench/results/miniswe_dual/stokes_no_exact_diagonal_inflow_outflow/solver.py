import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc
import sympy as sp

ScalarType = PETSc.ScalarType


def _parse_expression_ufl(expr_str, x_ufl):
    """Parse a string expression to UFL using sympy."""
    sx, sy, sz = sp.symbols("x y z", real=True)
    local_dict = {"x": sx, "y": sy, "z": sz}
    expr_sympy = sp.sympify(expr_str, locals=local_dict)
    return _sympy_to_ufl(expr_sympy, x_ufl)


def _sympy_to_ufl(expr, x):
    """Convert sympy expression to UFL."""
    sx, sy, sz = sp.symbols("x y z", real=True)
    
    if expr.is_Number:
        val = float(expr)
        if val == 0.0:
            return 0.0 * x[0]
        return ufl.as_ufl(val) * (1.0 + 0.0 * x[0])
    if expr.is_Symbol:
        if expr == sx:
            return x[0]
        if expr == sy:
            return x[1]
        if expr == sz:
            return x[2]
        raise ValueError(f"Unknown symbol: {expr}")
    if expr.func == sp.Add:
        result = _sympy_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result + _sympy_to_ufl(arg, x)
        return result
    if expr.func == sp.Mul:
        result = _sympy_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result * _sympy_to_ufl(arg, x)
        return result
    if expr.func == sp.Pow:
        base = _sympy_to_ufl(expr.args[0], x)
        exp_val = _sympy_to_ufl(expr.args[1], x)
        return base ** exp_val
    if expr.func == sp.sin:
        return ufl.sin(_sympy_to_ufl(expr.args[0], x))
    if expr.func == sp.cos:
        return ufl.cos(_sympy_to_ufl(expr.args[0], x))
    if expr.func == sp.exp:
        return ufl.exp(_sympy_to_ufl(expr.args[0], x))
    if expr == sp.pi:
        return float(sp.pi)
    raise NotImplementedError(f"Unsupported sympy expression: {expr} (type: {expr.func})")


def _boundary_selector(on_str, dim):
    """Create boundary marker function from string."""
    key = on_str.lower()
    if key in {"all", "*"}:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    if key in {"x0", "xmin"}:
        return lambda x: np.isclose(x[0], 0.0)
    if key in {"x1", "xmax"}:
        return lambda x: np.isclose(x[0], 1.0)
    if key in {"y0", "ymin"}:
        return lambda x: np.isclose(x[1], 0.0)
    if key in {"y1", "ymax"}:
        return lambda x: np.isclose(x[1], 1.0)
    raise ValueError(f"Unknown boundary selector: {on_str}")


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow with Taylor-Hood elements."""
    comm = MPI.COMM_WORLD
    
    # Parse configuration
    oracle_cfg = case_spec.get("oracle_config", case_spec)
    pde_cfg = oracle_cfg.get("pde", {})
    domain_cfg = oracle_cfg.get("domain", {"type": "unit_square"})
    bc_cfg = oracle_cfg.get("bc", {})
    output_cfg = oracle_cfg.get("output", {})
    
    # PDE parameters
    pde_params = pde_cfg.get("pde_params", {})
    nu_val = float(pde_params.get("nu", 0.8))
    source_term = pde_cfg.get("source_term", ["0.0", "0.0"])
    
    # Output grid
    grid_cfg = output_cfg.get("grid", {"bbox": [0, 1, 0, 1], "nx": 100, "ny": 100})
    nx_out = grid_cfg.get("nx", 100)
    ny_out = grid_cfg.get("ny", 100)
    bbox = grid_cfg.get("bbox", [0, 1, 0, 1])
    
    # Mesh resolution - use adaptive approach
    N = 64  # Good balance of accuracy and speed for this problem
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space (Taylor-Hood P2/P1)
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mixed_el)
    
    # Also create individual velocity space for BC interpolation
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    # Define variational problem
    u_trial = ufl.TrialFunction(W)
    v_test = ufl.TestFunction(W)
    (u, p) = ufl.split(u_trial)
    (v, q) = ufl.split(v_test)
    
    x_ufl = ufl.SpatialCoordinate(domain)
    
    # Source term
    try:
        f_const = [float(sp.sympify(s)) for s in source_term]
        f = fem.Constant(domain, ScalarType(np.array(f_const, dtype=np.float64)))
    except:
        f_components = [_parse_expression_ufl(s, x_ufl) for s in source_term]
        f = ufl.as_vector(f_components)
    
    # Bilinear form
    nu = fem.Constant(domain, ScalarType(nu_val))
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    
    # Build Dirichlet BCs
    bcs = []
    dirichlet_cfgs = bc_cfg.get("dirichlet", [])
    if isinstance(dirichlet_cfgs, dict):
        dirichlet_cfgs = [dirichlet_cfgs]
    
    for dc in dirichlet_cfgs:
        on = dc.get("on", "all")
        value = dc.get("value", ["0.0", "0.0"])
        
        selector = _boundary_selector(on, gdim)
        
        # Create BC function
        bc_func = fem.Function(V)
        
        if isinstance(value, (list, tuple)):
            expr_list = list(value)
        else:
            expr_list = [str(value)] * gdim
        
        # Check if all constant
        try:
            const_vals = [float(sp.sympify(e)) for e in expr_list]
            bc_func.interpolate(lambda x, cv=const_vals: np.array([[cv[i]] * x.shape[1] for i in range(len(cv))]))
        except (ValueError, TypeError):
            # Use UFL expression
            bc_components = [_parse_expression_ufl(e, x_ufl) for e in expr_list]
            bc_expr = ufl.as_vector(bc_components)
            interp_points = V.element.interpolation_points
            expr_compiled = fem.Expression(bc_expr, interp_points)
            bc_func.interpolate(expr_compiled)
        
        # Locate boundary DOFs
        boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V), selector)
        bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
    
    # Pressure fixing (pin pressure at origin)
    p_func = fem.Function(Q)
    p_func.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        bcs.append(fem.dirichletbc(p_func, p_dofs, W.sub(1)))
    
    # Solve linear problem
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10
    
    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options=petsc_options,
        petsc_options_prefix="stokes_"
    )
    
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Extract velocity
    u_h = wh.sub(0).collapse()
    
    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx_out)
    y_grid = np.linspace(ymin, ymax, ny_out)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_mag = np.full(points.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts, cls)
        mag = np.linalg.norm(vals, axis=1)
        vel_mag[eval_map] = mag
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Get iteration count
    try:
        ksp = problem.solver
        iterations = ksp.getIterationNumber()
    except:
        iterations = 0
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }


if __name__ == "__main__":
    import time
    import json
    
    # Load case spec
    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "stokes",
                "pde_params": {"nu": 0.8},
                "source_term": ["0.0", "0.0"]
            },
            "domain": {"type": "unit_square"},
            "bc": {
                "dirichlet": [
                    {"on": "x0", "value": ["2*y*(1-y)", "2*y*(1-y)"]},
                    {"on": "y0", "value": ["0.0", "0.0"]},
                    {"on": "y1", "value": ["0.0", "0.0"]}
                ]
            },
            "output": {
                "field": "velocity_magnitude",
                "grid": {"bbox": [0, 1, 0, 1], "nx": 100, "ny": 100}
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u = result["u"]
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {u.shape}")
    print(f"Solution range: [{np.nanmin(u):.6f}, {np.nanmax(u):.6f}]")
    print(f"Solution mean: {np.nanmean(u):.6f}")
    print(f"NaN count: {np.isnan(u).sum()}")
    print(f"Solver info: {result['solver_info']}")
    
    # Compare with reference
    try:
        ref = np.load('/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/results/miniswepde/stokes_no_exact_diagonal_inflow_outflow/oracle_output/reference.npz')
        u_ref = ref['u_star']
        mask = ~(np.isnan(u) | np.isnan(u_ref))
        diff = (u - u_ref)[mask]
        ref_vals = u_ref[mask]
        l2_diff = np.sqrt(np.sum(diff**2))
        l2_ref = np.sqrt(np.sum(ref_vals**2))
        rel_error = l2_diff / l2_ref if l2_ref > 1e-15 else l2_diff
        print(f"Relative L2 error vs reference: {rel_error:.6e}")
        print(f"Target: <= 4.86e-03")
        print(f"PASS: {rel_error <= 4.86e-3}")
    except Exception as e:
        print(f"Could not compare with reference: {e}")
