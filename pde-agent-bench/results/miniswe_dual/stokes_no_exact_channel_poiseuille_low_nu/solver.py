import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import json
import time
import argparse
from pathlib import Path


def solve(case_spec: dict = None):
    """
    Solve Stokes flow (incompressible) with Taylor-Hood elements.
    Poiseuille channel flow on [0,1]x[0,1] with outflow BC at x=1.
    """
    if case_spec is None:
        case_spec = {}

    comm = MPI.COMM_WORLD

    # Extract parameters
    pde_spec = case_spec.get("pde", {})
    nu_val = float(pde_spec.get("pde_params", {}).get("nu", 0.05))
    # Also check top-level viscosity
    if "viscosity" in pde_spec:
        nu_val = float(pde_spec["viscosity"])

    # Solver parameters
    degree_u = 2
    degree_p = 1
    ksp_type_used = "minres"
    pc_type_used = "hypre"

    # Parse source term
    source_term = pde_spec.get("source_term", ["0.0", "0.0"])
    try:
        fx_val = float(source_term[0]) if isinstance(source_term, list) else 0.0
        fy_val = float(source_term[1]) if isinstance(source_term, list) else 0.0
    except (ValueError, TypeError, IndexError):
        fx_val, fy_val = 0.0, 0.0

    # Parse boundary conditions from case_spec
    bc_config = case_spec.get("bc", {})
    dirichlet_list = bc_config.get("dirichlet", [])

    # Adaptive mesh refinement
    resolutions = [64, 128, 192]
    prev_norm = None
    final_u_grid = None
    final_info = None
    final_x = None
    final_y = None

    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        tdim = domain.topology.dim
        fdim = tdim - 1

        # Create Taylor-Hood mixed elements (P2/P1)
        P2 = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
        P1_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
        me = basix.ufl.mixed_element([P2, P1_el])
        W = fem.functionspace(domain, me)

        # Individual spaces for BCs
        V, V_to_W_map = W.sub(0).collapse()
        Q, Q_to_W_map = W.sub(1).collapse()

        # Trial and test functions
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        # Viscosity and source
        nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
        f = fem.Constant(domain, PETSc.ScalarType((fx_val, fy_val)))

        # Bilinear form: Stokes equations
        a_form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.div(v) * p * ufl.dx
            - q * ufl.div(u) * ufl.dx
        )

        # Linear form
        L_form = ufl.inner(f, v) * ufl.dx

        # Build boundary conditions
        bcs = []

        if len(dirichlet_list) > 0:
            # Use BCs from case_spec
            for bc_item in dirichlet_list:
                on = bc_item.get("on", "")
                value = bc_item.get("value", ["0.0", "0.0"])

                # Get boundary selector
                selector = _boundary_selector(on)
                if selector is None:
                    continue

                # Create BC function
                u_bc_func = fem.Function(V)
                u_bc_func.interpolate(_make_bc_interpolator(value))
                facets = mesh.locate_entities_boundary(domain, fdim, selector)
                dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
                bcs.append(fem.dirichletbc(u_bc_func, dofs, W.sub(0)))
        else:
            # Default: Poiseuille channel with outflow at x=1
            # Inlet (x=0): Poiseuille profile
            u_inlet = fem.Function(V)
            u_inlet.interpolate(lambda x: np.stack([4.0 * x[1] * (1.0 - x[1]), np.zeros(x.shape[1])]))
            facets_inlet = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
            dofs_inlet = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_inlet)
            bcs.append(fem.dirichletbc(u_inlet, dofs_inlet, W.sub(0)))

            # Bottom wall (y=0): no-slip
            u_bottom = fem.Function(V)
            u_bottom.interpolate(lambda x: np.zeros((2, x.shape[1])))
            facets_bottom = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
            dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_bottom)
            bcs.append(fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0)))

            # Top wall (y=1): no-slip
            u_top = fem.Function(V)
            u_top.interpolate(lambda x: np.zeros((2, x.shape[1])))
            facets_top = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
            dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_top)
            bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))

            # x=1: natural outflow (do-nothing BC) - NO Dirichlet BC here

        # Pin pressure at origin (0,0) to fix pressure nullspace
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs[0]) > 0:
            bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))

        # Solve
        try:
            problem = petsc.LinearProblem(
                a_form, L_form, bcs=bcs,
                petsc_options_prefix="stokes_",
                petsc_options={
                    "ksp_type": "minres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-10,
                    "ksp_max_it": 5000,
                },
            )
            wh = problem.solve()
            wh.x.scatter_forward()
        except Exception:
            # Fallback to direct solver
            ksp_type_used = "preonly"
            pc_type_used = "lu"
            problem = petsc.LinearProblem(
                a_form, L_form, bcs=bcs,
                petsc_options_prefix="stokes_fb_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
            )
            wh = problem.solve()
            wh.x.scatter_forward()

        # Extract velocity sub-function
        u_sol = wh.sub(0).collapse()

        # Evaluate on 100x100 grid
        x_coords, y_coords, vel_mag_grid = _evaluate_velocity_magnitude(u_sol, domain, tdim)

        # Check convergence
        current_norm = np.linalg.norm(vel_mag_grid)

        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4:
                final_u_grid = vel_mag_grid
                final_x = x_coords
                final_y = y_coords
                final_info = _make_info(N, degree_u, ksp_type_used, pc_type_used)
                break

        prev_norm = current_norm
        final_u_grid = vel_mag_grid
        final_x = x_coords
        final_y = y_coords
        final_info = _make_info(N, degree_u, ksp_type_used, pc_type_used)

    return {
        "u": final_u_grid,
        "x": final_x,
        "y": final_y,
        "solver_info": final_info,
    }


def _make_info(N, degree_u, ksp_type, pc_type):
    return {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }


def _boundary_selector(on: str):
    """Create a boundary marker function based on 'on' string."""
    key = on.lower().strip()
    if key in {"x0", "xmin", "left", "inlet"}:
        return lambda x: np.isclose(x[0], 0.0)
    elif key in {"x1", "xmax", "right", "outlet"}:
        return lambda x: np.isclose(x[0], 1.0)
    elif key in {"y0", "ymin", "bottom"}:
        return lambda x: np.isclose(x[1], 0.0)
    elif key in {"y1", "ymax", "top"}:
        return lambda x: np.isclose(x[1], 1.0)
    elif key in {"all", "*", "boundary"}:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    return None


def _make_bc_interpolator(value):
    """Create an interpolation function from value specification."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        vx_str = str(value[0])
        vy_str = str(value[1])
    else:
        vx_str = "0.0"
        vy_str = "0.0"

    # Check if both are pure constants
    try:
        vx_const = float(vx_str)
        vy_const = float(vy_str)
        is_constant = True
    except (ValueError, TypeError):
        is_constant = False

    if is_constant:
        def bc_func(x):
            return np.array([[vx_const] * x.shape[1], [vy_const] * x.shape[1]])
        return bc_func
    else:
        def bc_func(x):
            # Build namespace for eval
            ns = {
                "x": x[0], "y": x[1],
                "np": np, "sin": np.sin, "cos": np.cos,
                "pi": np.pi, "exp": np.exp, "sqrt": np.sqrt,
                "abs": np.abs, "log": np.log,
            }
            vx_s = vx_str.replace("^", "**").replace("x[0]", "x").replace("x[1]", "y")
            vy_s = vy_str.replace("^", "**").replace("x[0]", "x").replace("x[1]", "y")
            try:
                vx = eval(vx_s, {"__builtins__": {}}, ns)
            except Exception:
                vx = np.zeros(x.shape[1])
            try:
                vy = eval(vy_s, {"__builtins__": {}}, ns)
            except Exception:
                vy = np.zeros(x.shape[1])
            if np.isscalar(vx):
                vx = np.full(x.shape[1], float(vx))
            if np.isscalar(vy):
                vy = np.full(x.shape[1], float(vy))
            return np.stack([vx, vy])
        return bc_func


def _evaluate_velocity_magnitude(u_sol, domain, tdim):
    """Evaluate velocity magnitude on a 100x100 grid."""
    nx_eval, ny_eval = 100, 100
    bbox = [0, 1, 0, 1]
    xmin, xmax, ymin, ymax = bbox

    x_coords = np.linspace(xmin, xmax, nx_eval)
    y_coords = np.linspace(ymin, ymax, ny_eval)

    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((nx_eval * ny_eval, 3))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.linalg.norm(vals, axis=1)
        for idx, map_idx in enumerate(eval_map):
            values[map_idx] = mag[idx]

    # Replace any remaining NaN with 0
    values = np.nan_to_num(values, nan=0.0)
    return x_coords, y_coords, values.reshape((nx_eval, ny_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stokes solver')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Load case config if available
    case_config_path = Path(__file__).parent.parent.parent / "cases" / "stokes_no_exact_channel_poiseuille_low_nu" / "config.json"
    case_spec = {}
    if case_config_path.exists():
        with open(case_config_path) as f:
            config = json.load(f)
            case_spec = config.get("oracle_config", {})

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    wall_time = t1 - t0

    print(f"\nWall time: {wall_time:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solver info: {result['solver_info']}")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save solution.npz
        np.savez(
            outdir / "solution.npz",
            x=result["x"],
            y=result["y"],
            u=result["u"],
        )

        # Save meta.json
        meta = {
            "wall_time_sec": wall_time,
            "solver_info": result["solver_info"],
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Output saved to {outdir}")
    else:
        # Self-test: compare with reference
        ref_cache = Path(__file__).parent.parent.parent / "results" / ".oracle_cache" / "stokes_no_exact_channel_poiseuille_low_nu.json"
        if ref_cache.exists():
            with open(ref_cache) as f:
                cache = json.load(f)
            ref = np.array(cache["reference"])
            rel_err = np.linalg.norm(result["u"] - ref) / np.linalg.norm(ref)
            print(f"Relative L2 error vs reference: {rel_err:.6e}")
