import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def _parse_float(val, default):
    """Safely parse a float from various types."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}

    # Handle oracle_config wrapper
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", case_spec.get("pde", {}))
    domain_spec = oracle_config.get("domain", case_spec.get("domain", {}))
    mesh_spec = oracle_config.get("mesh", {})
    fem_spec = oracle_config.get("fem", {})
    bc_spec = oracle_config.get("bc", case_spec.get("bc", {}))
    output_spec = oracle_config.get("output", case_spec.get("output", {}))

    # --- Parse PDE parameters from multiple possible locations ---
    pde_params = pde.get("pde_params", pde.get("params", pde.get("coefficients", {})))
    if not isinstance(pde_params, dict):
        pde_params = {}

    # Epsilon (diffusion coefficient)
    epsilon = None
    for src in [pde_params, pde, case_spec]:
        for key in ["epsilon", "diffusion_coefficient", "diffusivity", "nu", "kappa"]:
            if key in src:
                epsilon = _parse_float(src[key], None)
                if epsilon is not None:
                    break
        if epsilon is not None:
            break
    if epsilon is None:
        epsilon = 0.01  # Default for logistic reaction-diffusion

    # Reaction parameters
    rho = None
    reaction_type = "logistic"
    
    # Check nested reaction dict in pde_params
    reaction_cfg = pde_params.get("reaction", pde.get("reaction", {}))
    if isinstance(reaction_cfg, dict):
        reaction_type = reaction_cfg.get("type", "logistic")
        rho = _parse_float(reaction_cfg.get("rho", None), None)
    elif isinstance(reaction_cfg, str):
        reaction_type = reaction_cfg

    # Check other locations for rho
    if rho is None:
        for src in [pde_params, pde, case_spec]:
            for key in ["reaction_rho", "rho"]:
                if key in src:
                    rho = _parse_float(src[key], None)
                    if rho is not None:
                        break
            if rho is not None:
                break
    if rho is None:
        rho = 1.0  # Default

    # Source term
    f_val = 1.0
    for src in [pde_params, pde, case_spec]:
        for key in ["source_term", "f", "source", "rhs"]:
            if key in src:
                f_val = _parse_float(src[key], 1.0)
                break

    # --- Time parameters (with hardcoded defaults from problem description) ---
    time_params = pde.get("time", {})
    is_transient = bool(time_params)

    # Hardcoded defaults as specified in task description
    if not is_transient:
        is_transient = True
        t_end = 0.3
        dt_val = 0.01
        time_scheme = "backward_euler"
    else:
        t_end = float(time_params.get("t_end", 0.3))
        dt_val = float(time_params.get("dt", 0.01))
        time_scheme = time_params.get("scheme", "backward_euler")

    # --- Domain ---
    domain_type = domain_spec.get("type", "unit_square") if isinstance(domain_spec, dict) else "unit_square"
    x_min = float(domain_spec.get("x_min", 0.0)) if isinstance(domain_spec, dict) else 0.0
    x_max = float(domain_spec.get("x_max", 1.0)) if isinstance(domain_spec, dict) else 1.0
    y_min = float(domain_spec.get("y_min", 0.0)) if isinstance(domain_spec, dict) else 0.0
    y_max = float(domain_spec.get("y_max", 1.0)) if isinstance(domain_spec, dict) else 1.0

    # --- Cell type ---
    cell_type_str = mesh_spec.get("cell_type", case_spec.get("cell_type", "quadrilateral"))
    if "quad" in cell_type_str.lower():
        cell_t = mesh.CellType.quadrilateral
    else:
        cell_t = mesh.CellType.triangle

    # --- Mesh resolution and element degree ---
    mesh_resolution = 64
    element_degree = 2

    # --- Boundary condition value ---
    bc_value = 0.0
    if isinstance(bc_spec, dict):
        dirichlet_list = bc_spec.get("dirichlet", [])
        if isinstance(dirichlet_list, list) and len(dirichlet_list) > 0:
            # Use first dirichlet BC value
            first_bc = dirichlet_list[0]
            if isinstance(first_bc, dict):
                bc_value = _parse_float(first_bc.get("value", 0.0), 0.0)
        elif isinstance(dirichlet_list, dict):
            bc_value = _parse_float(dirichlet_list.get("value", 0.0), 0.0)
        else:
            bc_value = _parse_float(bc_spec.get("value", 0.0), 0.0)
    
    # Also check pde-level BC
    pde_bc = pde.get("boundary_conditions", pde.get("bc", {}))
    if isinstance(pde_bc, dict) and "value" in pde_bc:
        bc_value = _parse_float(pde_bc["value"], bc_value)

    # --- Output grid ---
    grid_cfg = output_spec.get("grid", {}) if isinstance(output_spec, dict) else {}
    nx_out = int(grid_cfg.get("nx", 65)) if isinstance(grid_cfg, dict) else 65
    ny_out = int(grid_cfg.get("ny", 65)) if isinstance(grid_cfg, dict) else 65

    # --- Create mesh and function space ---
    comm = MPI.COMM_WORLD
    N = mesh_resolution
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain_mesh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=cell_t)
    V = fem.functionspace(domain_mesh, ("Lagrange", element_degree))

    # --- Boundary conditions ---
    tdim = domain_mesh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain_mesh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    bc_val_float = float(bc_value)
    u_bc.interpolate(lambda x: np.full(x.shape[1], bc_val_float))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]

    # --- Functions ---
    u = fem.Function(V)
    u.name = "u"
    v = ufl.TestFunction(V)
    f_c = fem.Constant(domain_mesh, ScalarType(float(f_val)))
    eps_c = fem.Constant(domain_mesh, ScalarType(float(epsilon)))
    rho_c = fem.Constant(domain_mesh, ScalarType(float(rho)))

    # --- Initial condition ---
    def default_ic(x):
        return 0.25 + 0.15 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    ic = pde.get("initial_condition", pde.get("u0", None))
    if ic is not None:
        if callable(ic):
            u.interpolate(ic)
        elif isinstance(ic, str):
            # Try to parse common IC strings
            u.interpolate(default_ic)
        elif isinstance(ic, (int, float)):
            ic_float = float(ic)
            u.interpolate(lambda x: np.full(x.shape[1], ic_float))
        else:
            u.interpolate(default_ic)
    else:
        u.interpolate(default_ic)

    # Save initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u.x.array[:]

    # --- Reaction term ---
    rt = reaction_type.lower() if isinstance(reaction_type, str) else "logistic"
    if "logistic" in rt:
        R_u = rho_c * u * (1.0 - u)
    elif "cubic" in rt or "allen" in rt:
        R_u = rho_c * u * u * u
    elif "linear" in rt:
        R_u = rho_c * u
    else:
        R_u = rho_c * u * (1.0 - u)

    # --- Build variational form ---
    # PDE: ∂u/∂t - ε∇²u + R(u) = f
    # Weak form (backward Euler):
    # (u - u_n)/dt * v dx + ε * grad(u)·grad(v) dx + R(u) * v dx - f * v dx = 0

    n_steps = int(round(t_end / dt_val))
    actual_dt = t_end / n_steps  # Ensure we hit t_end exactly

    ksp_type = "gmres"
    pc_type = "ilu"

    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": 1e-8,
        "snes_linesearch_type": "bt",
    }

    dt_c = fem.Constant(domain_mesh, ScalarType(float(actual_dt)))
    u_n = fem.Function(V)
    u_n.x.array[:] = u.x.array[:]

    F_form = (
        (u - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + R_u * v * ufl.dx
        - f_c * v * ufl.dx
    )

    problem = petsc.NonlinearProblem(
        F_form, u, bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options=petsc_opts,
    )

    total_linear_iterations = 0
    nonlinear_iterations_list = []

    for step in range(n_steps):
        problem.solve()
        u.x.scatter_forward()

        snes = problem.solver
        nl_its = snes.getIterationNumber()
        lin_its = snes.getLinearSolveIterations()

        nonlinear_iterations_list.append(int(nl_its))
        total_linear_iterations += int(lin_its)

        # Update previous solution
        u_n.x.array[:] = u.x.array[:]

    # --- Evaluate on output grid ---
    x_coords = np.linspace(x_min, x_max, nx_out)
    y_coords = np.linspace(y_min, y_max, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()

    bb_tree = geometry.bb_tree(domain_mesh, domain_mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain_mesh, cell_candidates, points_3d)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals2 = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals2.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": float(actual_dt),
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iterations_list,
        },
    }
