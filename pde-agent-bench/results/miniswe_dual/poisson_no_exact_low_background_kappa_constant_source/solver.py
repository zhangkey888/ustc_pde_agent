import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None):
    """
    Solve the Poisson equation: -div(kappa * grad(u)) = f on [0,1]^2
    with Dirichlet BC u = g on boundary.
    """
    comm = MPI.COMM_WORLD

    if case_spec is None:
        case_spec = {}

    # Extract PDE spec - try multiple locations
    pde = case_spec.get('pde', {})
    if not pde:
        # Try oracle_config.pde
        oracle_cfg = case_spec.get('oracle_config', {})
        pde = oracle_cfg.get('pde', {})

    # Extract source term
    source_val = 1.0
    source_spec = pde.get('source_term', pde.get('source', None))
    if source_spec is not None:
        if isinstance(source_spec, (int, float)):
            source_val = float(source_spec)
        elif isinstance(source_spec, str):
            try:
                source_val = float(source_spec)
            except ValueError:
                source_val = 1.0

    # Extract boundary condition value
    bc_val = 0.0
    # Try oracle_config.bc first
    oracle_cfg = case_spec.get('oracle_config', {})
    bc_cfg = oracle_cfg.get('bc', {})
    if bc_cfg:
        dirichlet_cfg = bc_cfg.get('dirichlet', {})
        if isinstance(dirichlet_cfg, dict):
            bc_val_str = dirichlet_cfg.get('value', '0.0')
            try:
                bc_val = float(bc_val_str)
            except (ValueError, TypeError):
                bc_val = 0.0
    else:
        # Try pde.bcs
        bcs_spec = pde.get('bcs', None)
        if bcs_spec is not None:
            if isinstance(bcs_spec, dict):
                bc_val = float(bcs_spec.get('value', 0.0))
            elif isinstance(bcs_spec, list):
                for bc_s in bcs_spec:
                    if isinstance(bc_s, dict):
                        bc_val = float(bc_s.get('value', 0.0))

    # Extract kappa expression
    kappa_expr_str = None
    kappa_const_val = 1.0
    coeffs = pde.get('coefficients', {})
    if isinstance(coeffs, dict):
        kappa_spec = coeffs.get('kappa', coeffs.get('κ', None))
        if isinstance(kappa_spec, dict):
            kappa_expr_str = kappa_spec.get('expr', None)
            if kappa_expr_str is None and 'value' in kappa_spec:
                kappa_const_val = float(kappa_spec['value'])
        elif isinstance(kappa_spec, (int, float)):
            kappa_const_val = float(kappa_spec)

    # Output grid
    nx_out, ny_out = 50, 50
    bbox = [0, 1, 0, 1]
    output_cfg = oracle_cfg.get('output', {})
    grid_cfg = output_cfg.get('grid', {})
    if grid_cfg:
        nx_out = grid_cfg.get('nx', 50)
        ny_out = grid_cfg.get('ny', 50)
        bbox = grid_cfg.get('bbox', [0, 1, 0, 1])

    # Solver parameters - adaptive mesh refinement
    resolutions = [48, 80, 128]
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    prev_norm = None
    u_sol = None
    domain = None
    final_N = resolutions[0]

    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))

        x = ufl.SpatialCoordinate(domain)

        # Define kappa
        if kappa_expr_str is not None:
            kappa = _parse_kappa_expr(kappa_expr_str, x, domain)
        else:
            kappa = fem.Constant(domain, ScalarType(kappa_const_val))

        # Source term
        f = fem.Constant(domain, ScalarType(source_val))

        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Bilinear and linear forms
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx

        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(bc_val), dofs, V)

        # Solve
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "2000",
                },
                petsc_options_prefix="poisson_"
            )
            uh = problem.solve()
        except Exception:
            ksp_type_used = "preonly"
            pc_type_used = "lu"
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type_used,
                    "pc_type": pc_type_used,
                },
                petsc_options_prefix="poisson_lu_"
            )
            uh = problem.solve()

        # Compute L2 norm for convergence check
        norm_val = np.sqrt(comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)),
            op=MPI.SUM
        ))

        u_sol = uh
        final_N = N

        if prev_norm is not None:
            rel_err = abs(norm_val - prev_norm) / (abs(norm_val) + 1e-15)
            if rel_err < 0.005:
                break

        prev_norm = norm_val

    # Get iteration count
    iterations = 0
    try:
        iterations = problem.solver.getIterationNumber()
    except Exception:
        pass

    # Evaluate on output grid
    xmin, xmax, ymin, ymax = bbox
    x_coords = np.linspace(xmin, xmax, nx_out)
    y_coords = np.linspace(ymin, ymax, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = xx.ravel()
    points_3d[:, 1] = yy.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        },
    }


def _parse_kappa_expr(expr_str, x, domain):
    """Parse a kappa expression string and return a UFL expression."""
    try:
        val = float(expr_str)
        return fem.Constant(domain, ScalarType(val))
    except ValueError:
        pass

    safe_ns = {
        'x': x[0],
        'y': x[1],
        'exp': ufl.exp,
        'sin': ufl.sin,
        'cos': ufl.cos,
        'sqrt': ufl.sqrt,
        'pi': np.pi,
        'log': ufl.ln,
        'ln': ufl.ln,
        'tanh': ufl.tanh,
        'cosh': ufl.cosh,
        'sinh': ufl.sinh,
    }

    try:
        kappa_ufl = eval(expr_str, {"__builtins__": {}}, safe_ns)
        return kappa_ufl
    except Exception as e:
        print(f"Warning: Could not parse kappa expression '{expr_str}': {e}")
        return fem.Constant(domain, ScalarType(1.0))


if __name__ == "__main__":
    import time

    case_spec = {
        "oracle_config": {
            "pde": {
                "type": "poisson",
                "coefficients": {
                    "kappa": {
                        "type": "expr",
                        "expr": "0.2 + 0.8*exp(-80*((x-0.5)**2 + (y-0.5)**2))"
                    }
                },
                "source_term": "1.0"
            },
            "bc": {
                "dirichlet": {"on": "all", "value": "0.0"}
            },
            "output": {
                "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50}
            }
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    info = result["solver_info"]

    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {info}")
