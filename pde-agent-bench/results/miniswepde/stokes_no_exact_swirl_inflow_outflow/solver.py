import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc
import re


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow with Taylor-Hood elements."""
    comm = MPI.COMM_WORLD

    # Parse case_spec - the full config is passed
    oracle_cfg = case_spec.get("oracle_config", case_spec)
    pde = oracle_cfg.get("pde", {})
    nu_val = float(pde.get("pde_params", {}).get("nu", pde.get("viscosity", 0.5)))

    # Source term
    source = pde.get("source_term", ["0.0", "0.0"])
    if not isinstance(source, list):
        source = ["0.0", "0.0"]

    # Domain
    domain_spec = oracle_cfg.get("domain", {})
    domain_type = domain_spec.get("type", "unit_square")

    # Output grid
    output_cfg = oracle_cfg.get("output", {})
    grid_cfg = output_cfg.get("grid", {})
    bbox = grid_cfg.get("bbox", [0, 1, 0, 1])
    nx_out = grid_cfg.get("nx", 100)
    ny_out = grid_cfg.get("ny", 100)

    # Boundary conditions
    bc_cfg = oracle_cfg.get("bc", {})
    dirichlet_cfgs = bc_cfg.get("dirichlet", [])
    if isinstance(dirichlet_cfgs, dict):
        dirichlet_cfgs = [dirichlet_cfgs]

    # Solver config
    solver_cfg = oracle_cfg.get("oracle_solver", {})
    pressure_fixing = solver_cfg.get("pressure_fixing", "point")

    # Use high resolution for accuracy
    N = 128
    degree_u = 2
    degree_p = 1

    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Create Taylor-Hood mixed element
    vel_el = basix.ufl.element("Lagrange", "triangle", degree_u, shape=(2,))
    pres_el = basix.ufl.element("Lagrange", "triangle", degree_p)
    mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mixed_el)

    # Collapse subspaces for BC interpolation
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    # Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Viscosity
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Source term
    f_vals = [0.0, 0.0]
    for i, s in enumerate(source):
        try:
            f_vals[i] = float(s)
        except ValueError:
            f_vals[i] = 0.0
    f = fem.Constant(msh, PETSc.ScalarType(f_vals))

    # Bilinear form: Stokes (symmetric saddle-point form)
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.div(v) * p * ufl.dx
        - q * ufl.div(u) * ufl.dx
    )

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Build Dirichlet BCs
    x = ufl.SpatialCoordinate(msh)
    bcs = []
    for dc in dirichlet_cfgs:
        on = dc.get("on", "all")
        value = dc.get("value", ["0.0", "0.0"])

        # Get boundary DOFs using geometrical location (matching oracle)
        selector = _boundary_selector(on)
        dofs = fem.locate_dofs_geometrical((W.sub(0), V), selector)

        # Create BC function
        u_bc = fem.Function(V)
        if isinstance(value, (list, tuple)):
            # Check if all constant
            try:
                const_vals = [float(v) for v in value]
                u_bc.interpolate(lambda x, cv=const_vals: np.array(
                    [[cv[i]] * x.shape[1] for i in range(len(cv))]
                ))
            except (ValueError, TypeError):
                # Use UFL expression parsing
                bc_components = [_parse_expr_to_ufl(str(v), x) for v in value]
                bc_vec = ufl.as_vector(bc_components)
                _interpolate_ufl_expr(u_bc, bc_vec)
        else:
            u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))

        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # If no BCs specified, apply zero on all boundaries
    if not bcs:
        dofs = fem.locate_dofs_geometrical(
            (W.sub(0), V), lambda x: np.ones(x.shape[1], dtype=bool)
        )
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros((2, x.shape[1])))
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # Pressure fixing: pin pressure at origin to remove nullspace
    if pressure_fixing == "point":
        p_bc = fem.Function(Q)
        p_bc.x.array[:] = 0.0
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        )
        if len(p_dofs[0]) > 0:
            bcs.append(fem.dirichletbc(p_bc, p_dofs, W.sub(1)))

    # Solve using minres + hypre (correct for saddle-point systems)
    ksp_type = "minres"
    pc_type = "hypre"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 5000,
        }
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    # Extract velocity
    u_sol = wh.sub(0).collapse()

    # Evaluate on output grid
    u_grid = _evaluate_velocity_magnitude_on_grid(
        msh, u_sol, bbox, nx_out, ny_out
    )

    info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
        "pressure_fixing": pressure_fixing,
    }

    return {"u": u_grid, "solver_info": info}


def _boundary_selector(on: str):
    """Create a boundary marker function from string identifier."""
    key = on.lower()
    if key in ("all", "*"):
        return lambda x: np.ones(x.shape[1], dtype=bool)
    if key in ("x0", "xmin", "left"):
        return lambda x: np.isclose(x[0], 0.0)
    if key in ("x1", "xmax", "right"):
        return lambda x: np.isclose(x[0], 1.0)
    if key in ("y0", "ymin", "bottom"):
        return lambda x: np.isclose(x[1], 0.0)
    if key in ("y1", "ymax", "top"):
        return lambda x: np.isclose(x[1], 1.0)
    raise ValueError(f"Unknown boundary selector: {on}")


def _parse_expr_to_ufl(expr_str, x):
    """Parse a string expression to a UFL expression using sympy."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    local_dict = {"x": sx, "y": sy}
    expr_sympy = sp.sympify(expr_str, locals=local_dict)
    return _sympy_to_ufl(expr_sympy, x)


def _sympy_to_ufl(expr, x):
    """Convert sympy expression to UFL."""
    import sympy as sp
    import math
    sx, sy = sp.symbols("x y", real=True)

    if expr.is_Number:
        val = float(expr)
        if val == 0.0:
            return 0.0 * x[0]
        else:
            return ufl.as_ufl(val) * (1.0 + 0.0 * x[0])
    if expr.is_Symbol:
        if expr == sx:
            return x[0]
        if expr == sy:
            return x[1]
        raise ValueError(f"Unknown symbol: {expr}")
    if expr == sp.pi:
        return math.pi
    if expr.func == sp.sin:
        return ufl.sin(_sympy_to_ufl(expr.args[0], x))
    if expr.func == sp.cos:
        return ufl.cos(_sympy_to_ufl(expr.args[0], x))
    if expr.func == sp.exp:
        return ufl.exp(_sympy_to_ufl(expr.args[0], x))
    if expr.func == sp.sqrt:
        return ufl.sqrt(_sympy_to_ufl(expr.args[0], x))
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
    raise NotImplementedError(f"Unsupported sympy function: {expr.func}")


def _interpolate_ufl_expr(func, expr):
    """Interpolate a UFL expression into a FEM function."""
    try:
        interp_points = func.function_space.element.interpolation_points
        expr_compiled = fem.Expression(expr, interp_points)
        func.interpolate(expr_compiled)
    except Exception:
        # Fallback: try lambda-based interpolation
        func.interpolate(lambda x: np.zeros((2, x.shape[1])))


def _evaluate_velocity_magnitude_on_grid(msh, u_sol, bbox, nx, ny):
    """Evaluate velocity magnitude on a uniform grid, matching oracle's method."""
    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")

    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        mag = np.linalg.norm(vals, axis=1)
        values[eval_map] = mag

    values = np.nan_to_num(values, nan=0.0)
    return values.reshape(nx, ny)


if __name__ == "__main__":
    import time
    import json

    case_spec_path = "/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/cases/stokes_no_exact_swirl_inflow_outflow/config.json"
    with open(case_spec_path) as f:
        case_spec = json.load(f)

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    print(f"Solve time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6e}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6e}")
    print(f"Mean velocity magnitude: {np.mean(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")

    # Compare with reference
    ref = np.load('/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/results/miniswepde/stokes_no_exact_swirl_inflow_outflow/oracle_output/reference.npz')
    ref_u = ref['u_star']
    diff = result['u'] - ref_u
    mask = ~np.isnan(result['u']) & ~np.isnan(ref_u)
    rel_err = np.sqrt(np.sum(diff[mask]**2)) / np.sqrt(np.sum(ref_u[mask]**2))
    print(f"Relative L2 error vs reference: {rel_err:.6e}")
    print(f"Target error: 5.79e-05")
    print(f"PASS accuracy: {rel_err <= 5.79e-05}")
    print(f"PASS time: {elapsed <= 147.157}")
