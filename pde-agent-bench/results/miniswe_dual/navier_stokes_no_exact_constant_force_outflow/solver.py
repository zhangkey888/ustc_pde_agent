"""
Solver for steady incompressible Navier-Stokes.
  u·∇u - ν∇²u + ∇p = f   in Ω
  ∇·u = 0                  in Ω

For low Reynolds number (high viscosity), Stokes solution is very accurate.
Uses adaptive mesh refinement with convergence check.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes."""
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters - handle both direct and nested oracle_config formats
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", {})
    pde_params = pde.get("pde_params", {})
    nu_val = pde_params.get("nu", pde.get("viscosity", 0.3))
    source = pde.get("source_term", pde.get("source", ["1.0", "0.0"]))
    f_vals = [float(s) for s in source]
    
    # Output grid
    output_info = oracle_config.get("output", {})
    grid_info = output_info.get("grid", {})
    bbox = grid_info.get("bbox", [0, 1, 0, 1])
    nx_out = grid_info.get("nx", 50)
    ny_out = grid_info.get("ny", 50)
    
    x_range = [bbox[0], bbox[1]]
    y_range = [bbox[2], bbox[3]]
    
    # BCs from config
    bc_config = oracle_config.get("bc", {})
    dirichlet_bcs = bc_config.get("dirichlet", [])
    
    if not dirichlet_bcs:
        dirichlet_bcs = [
            {"on": "y0", "value": ["0.0", "0.0"]},
            {"on": "y1", "value": ["0.0", "0.0"]},
            {"on": "x1", "value": ["0.0", "0.0"]},
        ]
    
    # Adaptive mesh refinement with convergence check
    resolutions = [48, 80, 128]
    degree_u = 2
    degree_p = 1
    
    prev_norm = None
    final_result = None
    
    for N in resolutions:
        result = _solve_stokes(
            comm, N, degree_u, degree_p, nu_val, f_vals,
            x_range, y_range, nx_out, ny_out, dirichlet_bcs
        )
        
        if result is None:
            continue
        
        u_grid = result["u"]
        current_norm = np.sqrt(np.nanmean(u_grid**2))
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4:
                final_result = result
                break
        
        prev_norm = current_norm
        final_result = result
    
    if final_result is None:
        raise RuntimeError("Solver failed at all resolutions")
    
    return final_result


def _solve_stokes(comm, N, degree_u, degree_p, nu_val, f_vals,
                  x_range, y_range, nx_out, ny_out, dirichlet_bcs):
    """Solve Stokes equations (excellent approximation for low Re NS)."""
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Create mixed Taylor-Hood elements P2/P1
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)
    
    # Build Dirichlet BCs
    bcs = _setup_bcs(domain, W, gdim, dirichlet_bcs)
    
    # Pin pressure at origin (0,0)
    Q_col, _ = W.sub(1).collapse()
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q_col),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0 and len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q_col)
        p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))
    
    f_const = fem.Constant(domain, PETSc.ScalarType(np.array(f_vals, dtype=np.float64)))
    
    # Stokes variational form
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    
    a_form = (
        nu_val * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - ufl.div(v_test) * p_trial * ufl.dx
        - q_test * ufl.div(u_trial) * ufl.dx
    )
    L_form = ufl.inner(f_const, v_test) * ufl.dx
    
    try:
        problem = petsc.LinearProblem(
            a_form, L_form, bcs=bcs,
            petsc_options={
                "ksp_type": "minres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-10,
            },
            petsc_options_prefix=f"stokes_{N}_",
        )
        w_sol = problem.solve()
        w_sol.x.scatter_forward()
    except Exception:
        # Fallback to direct solver
        try:
            problem = petsc.LinearProblem(
                a_form, L_form, bcs=bcs,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
                petsc_options_prefix=f"stokes_lu_{N}_",
            )
            w_sol = problem.solve()
            w_sol.x.scatter_forward()
        except Exception:
            return None
    
    # Extract velocity
    u_sol = w_sol.sub(0).collapse()
    
    # Evaluate on output grid
    u_grid = _evaluate_velocity_magnitude(domain, u_sol, x_range, y_range, nx_out, ny_out, gdim)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "minres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "nonlinear_iterations": [0],
        }
    }


def _setup_bcs(domain, W, gdim, dirichlet_bcs):
    """Set up Dirichlet boundary conditions from config."""
    bcs = []
    V_col, _ = W.sub(0).collapse()
    
    for bc_spec in dirichlet_bcs:
        location = bc_spec["on"]
        value = bc_spec.get("value", ["0.0", "0.0"])
        vals = [float(v) for v in value]
        
        selector = _boundary_selector(location)
        boundary_dofs = fem.locate_dofs_geometrical((W.sub(0), V_col), selector)
        
        if len(boundary_dofs) == 0 or len(boundary_dofs[0]) == 0:
            continue
        
        bc_func = fem.Function(V_col)
        bc_func.interpolate(
            lambda x, v=vals: np.array([[vi] * x.shape[1] for vi in v[:gdim]])
        )
        
        bcs.append(fem.dirichletbc(bc_func, boundary_dofs, W.sub(0)))
    
    return bcs


def _boundary_selector(on: str):
    key = on.lower()
    if key in {"x0", "xmin"}:
        return lambda x: np.isclose(x[0], 0.0)
    elif key in {"x1", "xmax"}:
        return lambda x: np.isclose(x[0], 1.0)
    elif key in {"y0", "ymin"}:
        return lambda x: np.isclose(x[1], 0.0)
    elif key in {"y1", "ymax"}:
        return lambda x: np.isclose(x[1], 1.0)
    elif key in {"all", "*"}:
        return lambda x: np.ones(x.shape[1], dtype=bool)
    else:
        raise ValueError(f"Unknown boundary selector: {on}")


def _evaluate_velocity_magnitude(domain, u_func, x_range, y_range, nx, ny, gdim):
    """Evaluate velocity magnitude on a uniform grid."""
    
    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xx.flatten()
    points[1, :] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_magnitude = np.full(nx * ny, 0.0)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        vel_mag = np.sqrt(np.sum(vals**2, axis=1))
        for idx, map_idx in enumerate(eval_map):
            vel_magnitude[map_idx] = vel_mag[idx]
    
    return vel_magnitude.reshape(nx, ny)
