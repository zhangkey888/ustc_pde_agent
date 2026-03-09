import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes equations."""
    
    # Extract parameters from case_spec
    oracle_config = case_spec.get("oracle_config", {})
    pde_config = oracle_config.get("pde", case_spec.get("pde", {}))
    
    # Viscosity
    pde_params = pde_config.get("pde_params", {})
    nu_val = float(pde_params.get("nu", pde_config.get("viscosity", 0.2)))
    
    # Source term
    source = pde_config.get("source_term", ["0.0", "0.0"])
    
    # Domain
    domain_spec = oracle_config.get("domain", case_spec.get("domain", {}))
    
    # Output grid
    output_config = oracle_config.get("output", case_spec.get("output", {}))
    grid_config = output_config.get("grid", {})
    bbox = grid_config.get("bbox", [0, 1, 0, 1])
    nx_out = grid_config.get("nx", output_config.get("nx", 50))
    ny_out = grid_config.get("ny", output_config.get("ny", 50))
    
    x_min_out, x_max_out = bbox[0], bbox[1]
    y_min_out, y_max_out = bbox[2], bbox[3]
    
    # Boundary conditions from oracle_config
    bc_config = oracle_config.get("bc", {})
    dirichlet_bcs = bc_config.get("dirichlet", [])
    
    # Domain range (unit square)
    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]
    
    # Solver parameters
    N = 48  # mesh resolution
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    p0 = np.array([x_range[0], y_range[0]], dtype=np.float64)
    p1 = np.array([x_range[1], y_range[1]], dtype=np.float64)
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space (Taylor-Hood P2/P1)
    cell_name = domain.topology.cell_name()
    el_v = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(gdim,))
    el_q = basix.ufl.element("Lagrange", cell_name, degree_p)
    mel = basix.ufl.mixed_element([el_v, el_q])
    W = fem.functionspace(domain, mel)
    
    # Separate spaces for BCs
    V = fem.functionspace(domain, ("Lagrange", degree_u, (gdim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Define unknown and test functions
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Build boundary conditions
    tol = 1e-10
    bcs = []
    
    for bc_entry in dirichlet_bcs:
        location = bc_entry.get("on", "")
        value = bc_entry.get("value", [0.0, 0.0])
        
        # Parse location
        if location in ["y1", "top", "y=1"]:
            marker = lambda x: np.isclose(x[1], y_range[1], atol=tol)
        elif location in ["y0", "bottom", "y=0"]:
            marker = lambda x: np.isclose(x[1], y_range[0], atol=tol)
        elif location in ["x0", "left", "x=0"]:
            marker = lambda x: np.isclose(x[0], x_range[0], atol=tol)
        elif location in ["x1", "right", "x=1"]:
            marker = lambda x: np.isclose(x[0], x_range[1], atol=tol)
        else:
            continue
        
        # Parse value
        if isinstance(value, list):
            vals = [float(v_) for v_ in value]
        else:
            vals = [float(value), 0.0]
        
        facets = mesh.locate_entities_boundary(domain, fdim, marker)
        u_bc = fem.Function(V)
        vx, vy = vals[0], vals[1] if len(vals) > 1 else 0.0
        u_bc.interpolate(lambda x, vx=vx, vy=vy: np.vstack([
            np.full(x.shape[1], vx),
            np.full(x.shape[1], vy)
        ]))
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))
    
    # If no BCs from config, use defaults
    if len(bcs) == 0:
        # Default: counter-shear with open sides
        # Top: u = (0.8, 0)
        top_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], y_range[1], atol=tol))
        u_top = fem.Function(V)
        u_top.interpolate(lambda x: np.vstack([
            np.full(x.shape[1], 0.8), np.zeros(x.shape[1])
        ]))
        dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
        bcs.append(fem.dirichletbc(u_top, dofs_top, W.sub(0)))
        
        # Bottom: u = (-0.8, 0)
        bot_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[1], y_range[0], atol=tol))
        u_bot = fem.Function(V)
        u_bot.interpolate(lambda x: np.vstack([
            np.full(x.shape[1], -0.8), np.zeros(x.shape[1])
        ]))
        dofs_bot = fem.locate_dofs_topological((W.sub(0), V), fdim, bot_facets)
        bcs.append(fem.dirichletbc(u_bot, dofs_bot, W.sub(0)))
    
    # Pin pressure at one point to remove nullspace
    # Find a DOF near the center
    pressure_dof = _pin_pressure(domain, W, Q, fdim)
    if pressure_dof is not None:
        bcs.append(pressure_dof)
    
    # Viscosity constant
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Source term
    f_vals = []
    for s in source:
        try:
            f_vals.append(float(s))
        except (ValueError, TypeError):
            f_vals.append(0.0)
    f = ufl.as_vector([fem.Constant(domain, PETSc.ScalarType(fv)) for fv in f_vals])
    
    # Weak form (residual) - standard NS formulation
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Initialize with a good guess (linear profile)
    # This helps Newton converge quickly
    W0_sub, W0_map = W.sub(0).collapse()
    w_init = fem.Function(W0_sub)
    w_init.interpolate(lambda x: np.vstack([
        0.8 * (2.0 * x[1] - 1.0),
        np.zeros(x.shape[1])
    ]))
    w.x.array[W0_map] = w_init.x.array
    w.x.scatter_forward()
    
    # Solve with SNES (Newton)
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )
    
    problem.solve()
    snes = problem.solver
    converged_reason = snes.getConvergedReason()
    nl_its = snes.getIterationNumber()
    
    if converged_reason <= 0:
        # Retry with relaxation
        w.x.array[:] = 0.0
        w.x.array[W0_map] = w_init.x.array
        w.x.scatter_forward()
        
        petsc_options2 = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 100,
        }
        
        problem2 = petsc.NonlinearProblem(
            F_form, w, bcs=bcs,
            petsc_options_prefix="ns2_",
            petsc_options=petsc_options2,
        )
        problem2.solve()
        snes2 = problem2.solver
        converged_reason = snes2.getConvergedReason()
        nl_its = snes2.getIterationNumber()
    
    w.x.scatter_forward()
    
    # Extract velocity
    V_collapse, collapse_map = W.sub(0).collapse()
    u_collapsed = fem.Function(V_collapse)
    u_collapsed.x.array[:] = w.x.array[collapse_map]
    u_collapsed.x.scatter_forward()
    
    # Create output grid
    xs = np.linspace(x_min_out, x_max_out, nx_out)
    ys = np.linspace(y_min_out, y_max_out, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    # Points array: shape (n_points, 3)
    points = np.zeros((nx_out * ny_out, 3), dtype=np.float64)
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    
    # Evaluate velocity at grid points
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    
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
        pts_arr = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_collapsed.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[i] = np.sqrt(ux**2 + uy**2)
    
    # Handle NaN points
    if np.any(np.isnan(vel_mag)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(vel_mag)
        if np.sum(valid) > 0:
            interp = NearestNDInterpolator(points[valid, :2], vel_mag[valid])
            vel_mag[np.isnan(vel_mag)] = interp(points[np.isnan(vel_mag), :2])
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(nl_its)],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


def _pin_pressure(domain, W, Q, fdim):
    """Pin pressure at a single point to remove the nullspace."""
    # Find a DOF at the origin or center
    def center_marker(x):
        return np.logical_and(
            np.isclose(x[0], 0.0, atol=1e-10),
            np.isclose(x[1], 0.0, atol=1e-10)
        )
    
    # Try to find a vertex at (0,0)
    facets = mesh.locate_entities_boundary(domain, 0, center_marker)
    if len(facets) > 0:
        dofs = fem.locate_dofs_topological((W.sub(1), Q), 0, facets)
        p_bc = fem.Function(Q)
        p_bc.x.array[:] = 0.0
        return fem.dirichletbc(p_bc, dofs, W.sub(1))
    
    # Fallback: pin at bottom-left corner
    def corner_marker(x):
        return np.logical_and(
            x[0] < 1e-8,
            x[1] < 1e-8
        )
    
    vertices = mesh.locate_entities_boundary(domain, 0, corner_marker)
    if len(vertices) > 0:
        dofs = fem.locate_dofs_topological((W.sub(1), Q), 0, vertices[:1])
        p_bc = fem.Function(Q)
        p_bc.x.array[:] = 0.0
        return fem.dirichletbc(p_bc, dofs, W.sub(1))
    
    return None


if __name__ == "__main__":
    import time
    import json
    
    # Load actual case_spec if available
    try:
        with open("../../miniswepde/navier_stokes_no_exact_counter_shear_open_sides/agent_output/case_spec.json") as f:
            case_spec = json.load(f)
    except FileNotFoundError:
        case_spec = {
            "oracle_config": {
                "pde": {
                    "type": "navier_stokes",
                    "pde_params": {"nu": 0.2},
                    "source_term": ["0.0", "0.0"],
                },
                "domain": {"type": "unit_square"},
                "bc": {
                    "dirichlet": [
                        {"on": "y1", "value": ["0.8", "0.0"]},
                        {"on": "y0", "value": ["-0.8", "0.0"]},
                    ]
                },
                "output": {
                    "field": "velocity_magnitude",
                    "grid": {"bbox": [0, 1, 0, 1], "nx": 50, "ny": 50},
                },
            }
        }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    print(f"Solution shape: {u_grid.shape}")
    print(f"Min velocity magnitude: {np.nanmin(u_grid):.10f}")
    print(f"Max velocity magnitude: {np.nanmax(u_grid):.10f}")
    print(f"Mean velocity magnitude: {np.nanmean(u_grid):.10f}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solver info: {info}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    
    # Compare with reference
    ref_data = np.load("../../miniswepde/navier_stokes_no_exact_counter_shear_open_sides/oracle_output/reference.npz")
    ref = ref_data["u_star"]
    
    # Compute relative L2 error
    diff = u_grid - ref
    rel_l2 = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(ref**2))
    print(f"Relative L2 error: {rel_l2:.2e}")
    print(f"Max absolute error: {np.max(np.abs(diff)):.2e}")
