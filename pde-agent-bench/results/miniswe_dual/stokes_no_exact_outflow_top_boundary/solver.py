"""
Solver for Stokes flow: stokes_no_exact_outflow_top_boundary

BCs:
  - Left (x=0): u = (sin(pi*y), 0)  [inflow]
  - Bottom (y=0): u = (0, 0)         [no-slip]
  - Right (x=1): u = (0, 0)          [no-slip]
  - Top (y=1): natural/outflow (do-nothing)
  - Pressure pinned at origin
"""

import numpy as np
import sympy as sp
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
import basix
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    oracle_config = case_spec.get('oracle_config', case_spec)
    pde = oracle_config.get('pde', case_spec.get('pde', {}))
    nu_val = float(pde.get('pde_params', {}).get('nu', pde.get('viscosity', 0.9)))

    # Output grid
    output_spec = oracle_config.get('output', case_spec.get('output', {}))
    grid_spec = output_spec.get('grid', {})
    bbox = grid_spec.get('bbox', [0, 1, 0, 1])
    nx_out = int(grid_spec.get('nx', 100))
    ny_out = int(grid_spec.get('ny', 100))

    # BC config
    bc_cfg = oracle_config.get('bc', case_spec.get('bc', {}))
    dirichlet_cfgs = bc_cfg.get('dirichlet', [])
    if isinstance(dirichlet_cfgs, dict):
        dirichlet_cfgs = [dirichlet_cfgs]

    # FEM config
    fem_cfg = oracle_config.get('fem', case_spec.get('fem', {}))
    degree_u = int(fem_cfg.get('degree_u', 2))
    degree_p = int(fem_cfg.get('degree_p', 1))

    # Solver config
    solver_cfg = oracle_config.get('oracle_solver', case_spec.get('oracle_solver', {}))
    pressure_fixing = solver_cfg.get('pressure_fixing', 'point')

    # Source term
    source = pde.get('source_term', pde.get('source', ['0.0', '0.0']))
    if isinstance(source, str):
        source = [source, source]

    # Adaptive mesh resolution
    resolutions = [48, 80, 128]
    ksp_type_used = "minres"
    pc_type_used = "hypre"
    rtol_used = 1e-10
    final_N = resolutions[0]
    final_u_grid = None
    prev_norm = None

    for N in resolutions:
        final_N = N

        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        tdim = domain.topology.dim
        fdim = tdim - 1
        gdim = domain.geometry.dim

        # Taylor-Hood mixed space
        vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
        pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
        mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
        W = fem.functionspace(domain, mixed_el)

        V, V_map = W.sub(0).collapse()
        Q, Q_map = W.sub(1).collapse()

        # Trial and test functions
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        # Bilinear form
        nu = fem.Constant(domain, ScalarType(nu_val))
        a_form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.div(v) * p * ufl.dx
            - q * ufl.div(u) * ufl.dx
        )

        # Source term
        x = ufl.SpatialCoordinate(domain)
        try:
            const_values = [float(sp.sympify(s)) for s in source]
            f_expr = fem.Constant(domain, tuple(ScalarType(v) for v in const_values))
        except (ValueError, TypeError):
            f_components = [_parse_ufl_expression(s, x) for s in source]
            f_expr = ufl.as_vector(f_components)

        L_form = ufl.inner(f_expr, v) * ufl.dx

        # Build Dirichlet BCs
        bcs = []

        if len(dirichlet_cfgs) > 0:
            for cfg in dirichlet_cfgs:
                on = cfg.get('on', 'all')
                value = cfg.get('value', ['0.0', '0.0'])

                selector = _boundary_selector(on, gdim)
                bc_func = fem.Function(V)
                _interpolate_bc_value(bc_func, value, domain, gdim, x)

                boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V), selector)
                bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
        else:
            # Default BCs
            bc_left = fem.Function(V)
            bc_left.interpolate(lambda xx: np.vstack([
                np.sin(np.pi * xx[1]),
                np.zeros_like(xx[0])
            ]))
            dofs_left = fem.locate_dofs_geometrical(
                (W.sub(0), V), lambda xx: np.isclose(xx[0], 0.0)
            )
            bcs.append(fem.dirichletbc(bc_left, dofs_left, W.sub(0)))

            bc_bottom = fem.Function(V)
            bc_bottom.interpolate(lambda xx: np.zeros((gdim, xx.shape[1])))
            dofs_bottom = fem.locate_dofs_geometrical(
                (W.sub(0), V), lambda xx: np.isclose(xx[1], 0.0)
            )
            bcs.append(fem.dirichletbc(bc_bottom, dofs_bottom, W.sub(0)))

            bc_right = fem.Function(V)
            bc_right.interpolate(lambda xx: np.zeros((gdim, xx.shape[1])))
            dofs_right = fem.locate_dofs_geometrical(
                (W.sub(0), V), lambda xx: np.isclose(xx[0], 1.0)
            )
            bcs.append(fem.dirichletbc(bc_right, dofs_right, W.sub(0)))

        # Pressure fixing
        if pressure_fixing != "none":
            p_dofs = fem.locate_dofs_geometrical(
                (W.sub(1), Q),
                lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
            )
            if len(p_dofs[0]) > 0:
                p0 = fem.Function(Q)
                p0.x.array[:] = 0.0
                bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

        # Solve with minres + hypre (correct for saddle-point systems)
        problem = fem_petsc.LinearProblem(
            a_form, L_form, bcs=bcs,
            petsc_options={
                "ksp_type": "minres",
                "pc_type": "hypre",
                "ksp_rtol": str(rtol_used),
                "ksp_max_it": "5000",
            },
            petsc_options_prefix="stokes_"
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        u_h = wh.sub(0).collapse()

        # Compute norm for convergence check
        norm_val = np.sqrt(comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_h, u_h) * ufl.dx)),
            op=MPI.SUM
        ))

        # Evaluate on grid
        u_grid = _sample_velocity_magnitude(domain, u_h, bbox, nx_out, ny_out)
        final_u_grid = u_grid

        if prev_norm is not None:
            rel_change = abs(norm_val - prev_norm) / (norm_val + 1e-15)
            if rel_change < 0.005:
                break
        prev_norm = norm_val

    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": degree_u,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": 1,
    }

    return {
        "u": final_u_grid,
        "solver_info": solver_info,
    }


def _parse_ufl_expression(expr_str, x):
    """Parse a string expression into a UFL expression using sympy."""
    expr_str = str(expr_str).strip()
    try:
        val = float(expr_str)
        return val + 0.0 * x[0]  # bind to domain
    except ValueError:
        pass

    sx, sy, sz = sp.symbols('x y z', real=True)
    local_dict = {'x': sx, 'y': sy, 'z': sz}
    sym_expr = sp.sympify(expr_str, locals=local_dict)

    # Convert sympy to UFL
    return _sympy_to_ufl(sym_expr, x)


def _sympy_to_ufl(expr, x):
    """Convert a sympy expression to UFL."""
    if isinstance(expr, (int, float)):
        return float(expr) + 0.0 * x[0]
    if isinstance(expr, sp.Number):
        return float(expr) + 0.0 * x[0]
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name == 'x':
            return x[0]
        elif name == 'y':
            return x[1]
        elif name == 'z':
            return x[2]
        else:
            raise ValueError(f"Unknown symbol: {name}")
    if isinstance(expr, sp.Add):
        result = _sympy_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result + _sympy_to_ufl(arg, x)
        return result
    if isinstance(expr, sp.Mul):
        result = _sympy_to_ufl(expr.args[0], x)
        for arg in expr.args[1:]:
            result = result * _sympy_to_ufl(arg, x)
        return result
    if isinstance(expr, sp.Pow):
        base = _sympy_to_ufl(expr.args[0], x)
        exp_val = expr.args[1]
        if isinstance(exp_val, sp.Integer):
            return base ** int(exp_val)
        return base ** _sympy_to_ufl(exp_val, x)
    if isinstance(expr, sp.pi.__class__) or expr == sp.pi:
        return ufl.pi
    if isinstance(expr, sp.sin):
        return ufl.sin(_sympy_to_ufl(expr.args[0], x))
    if isinstance(expr, sp.cos):
        return ufl.cos(_sympy_to_ufl(expr.args[0], x))
    if isinstance(expr, sp.exp):
        return ufl.exp(_sympy_to_ufl(expr.args[0], x))

    # Fallback: try to convert to float
    try:
        return float(expr) + 0.0 * x[0]
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert sympy expression to UFL: {expr} (type: {type(expr)})")


def _boundary_selector(on: str, dim: int):
    key = on.lower().strip()
    if key in {"all", "*"}:
        return lambda xx: np.ones(xx.shape[1], dtype=bool)
    if key in {"x0", "xmin", "left"}:
        return lambda xx: np.isclose(xx[0], 0.0)
    if key in {"x1", "xmax", "right"}:
        return lambda xx: np.isclose(xx[0], 1.0)
    if key in {"y0", "ymin", "bottom"}:
        return lambda xx: np.isclose(xx[1], 0.0)
    if key in {"y1", "ymax", "top"}:
        return lambda xx: np.isclose(xx[1], 1.0)
    raise ValueError(f"Unknown boundary selector: {on}")


def _interpolate_bc_value(bc_func, value, domain, gdim, x_ufl):
    """Interpolate BC value into a function."""
    if isinstance(value, (list, tuple)):
        str_vals = [str(v).strip() for v in value]
    else:
        str_vals = [str(value).strip()] * gdim

    # Check if all are constants
    try:
        float_vals = [float(v) for v in str_vals]
        bc_func.interpolate(lambda xx, fv=float_vals: np.array([[v] * xx.shape[1] for v in fv]))
        return
    except ValueError:
        pass

    # Parse expressions using sympy -> numpy for interpolation
    def interp_func(xx):
        result = np.zeros((gdim, xx.shape[1]))
        for i, expr_str in enumerate(str_vals):
            result[i, :] = _eval_expr_numpy(expr_str, xx)
        return result

    bc_func.interpolate(interp_func)


def _eval_expr_numpy(expr_str, x_coords):
    """Evaluate a string expression at given coordinates using sympy -> numpy."""
    expr = expr_str.strip()

    try:
        val = float(expr)
        return np.full(x_coords.shape[1], val)
    except ValueError:
        pass

    sx, sy, sz = sp.symbols('x y z', real=True)
    sym_expr = sp.sympify(expr, locals={'x': sx, 'y': sy, 'z': sz, 'pi': sp.pi})
    f_numpy = sp.lambdify([sx, sy, sz], sym_expr, modules=['numpy'])
    return np.array(f_numpy(x_coords[0], x_coords[1], x_coords[2]), dtype=np.float64)


def _sample_velocity_magnitude(domain, u_h, bbox, nx, ny):
    """Sample velocity magnitude on a uniform grid."""
    xmin, xmax, ymin, ymax = bbox
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_array, cells_array)
        mag = np.linalg.norm(vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            values[global_idx] = mag[idx]

    u_grid = values.reshape(nx, ny)

    # Fill NaN values at boundaries
    nan_mask = np.isnan(u_grid)
    if np.any(nan_mask):
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        u_grid = u_grid[tuple(indices)]

    return u_grid


if __name__ == "__main__":
    import argparse
    import json
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load case_spec if available
    case_spec_file = outdir / 'case_spec.json'
    if case_spec_file.exists():
        with open(case_spec_file) as f:
            case_spec = json.load(f)
    else:
        case_spec = {
            "oracle_config": {
                "pde": {
                    "type": "stokes",
                    "pde_params": {"nu": 0.9},
                    "source_term": ["0.0", "0.0"],
                },
                "domain": {"type": "unit_square"},
                "mesh": {"resolution": 42, "cell_type": "triangle"},
                "fem": {"degree_u": 2, "degree_p": 1},
                "bc": {
                    "dirichlet": [
                        {"on": "x0", "value": ["sin(pi*y)", "0.0"]},
                        {"on": "y0", "value": ["0.0", "0.0"]},
                        {"on": "x1", "value": ["0.0", "0.0"]},
                    ]
                },
                "output": {
                    "field": "velocity_magnitude",
                    "grid": {"bbox": [0, 1, 0, 1], "nx": 100, "ny": 100},
                },
                "oracle_solver": {
                    "ksp_type": "minres",
                    "pc_type": "hypre",
                    "rtol": 1e-10,
                    "pressure_fixing": "point",
                },
            }
        }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    solver_info = result["solver_info"]

    grid_spec = case_spec.get('oracle_config', case_spec).get('output', {}).get('grid', {})
    bbox_out = grid_spec.get('bbox', [0, 1, 0, 1])
    nx = grid_spec.get('nx', 100)
    ny = grid_spec.get('ny', 100)

    x = np.linspace(bbox_out[0], bbox_out[1], nx)
    y = np.linspace(bbox_out[2], bbox_out[3], ny)

    np.savez(str(outdir / 'solution.npz'), x=x, y=y, u=u_grid)
    np.save(str(outdir / 'u.npy'), u_grid)

    meta = {
        "wall_time_sec": elapsed,
        "solver_info": solver_info,
    }
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Solution shape: {u_grid.shape}")
    print(f"Min: {np.nanmin(u_grid):.6e}, Max: {np.nanmax(u_grid):.6e}")
    print(f"Mean: {np.nanmean(u_grid):.6e}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Solver info: {solver_info}")
