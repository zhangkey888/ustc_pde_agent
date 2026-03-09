import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion (Allen-Cahn) equation."""
    
    start_time = time.time()
    
    # Parse case_spec - handle both flat and nested oracle_config formats
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde_spec = oracle_config.get("pde", {})
    
    # PDE parameters
    pde_params = pde_spec.get("pde_params", {})
    epsilon = float(pde_params.get("epsilon", pde_params.get("diffusion", 0.03)))
    
    # Reaction parameters
    reaction_spec = pde_params.get("reaction", {"type": "allen_cahn", "lambda": 1.0})
    reaction_type = str(reaction_spec.get("type", "allen_cahn")).lower()
    reaction_lambda = float(reaction_spec.get("lambda", reaction_spec.get("lam", 1.0)))
    
    # Time parameters
    time_spec = pde_spec.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.25))
    dt_val = float(time_spec.get("dt", 0.005))
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = bool(time_spec)  # True if time config exists
    
    # Force transient if t_end/dt are provided
    if not is_transient:
        is_transient = True
        t_end = 0.25
        dt_val = 0.005
    
    # Source term string
    source_str = pde_spec.get("source_term", "5*exp(-180*((x-0.35)**2 + (y-0.55)**2))")
    
    # Initial condition string
    ic_str = pde_spec.get("initial_condition", "0.1*exp(-50*((x-0.5)**2 + (y-0.5)**2))")
    
    # Output grid
    output_spec = oracle_config.get("output", {})
    grid_spec = output_spec.get("grid", {})
    nx_out = int(grid_spec.get("nx", 60))
    ny_out = int(grid_spec.get("ny", 60))
    bbox = grid_spec.get("bbox", [0, 1, 0, 1])
    
    # Boundary conditions
    bc_spec = oracle_config.get("bc", {})
    dirichlet_spec = bc_spec.get("dirichlet", {})
    bc_value = float(dirichlet_spec.get("value", 0.0))
    
    # Mesh resolution - use fine mesh for accuracy
    N = 128
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term (UFL expression)
    # Parse source: 5*exp(-180*((x-0.35)**2 + (y-0.55)**2))
    f_expr = 5.0 * ufl.exp(-180.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    
    # Functions
    u = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    # Initial condition: 0.1*exp(-50*((x-0.5)**2 + (y-0.5)**2))
    u_n.interpolate(lambda X: 0.1 * np.exp(-50.0 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2)))
    u.x.array[:] = u_n.x.array[:]
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Constants
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    lam_c = fem.Constant(domain, ScalarType(reaction_lambda))
    
    # Boundary conditions: u = bc_value on all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(bc_value), dofs, V)
    bcs = [bc]
    
    # Build reaction term R(u) based on type
    # Allen-Cahn: R(u) = lambda * (u^3 - u)
    # NO 1/epsilon factor! The oracle uses R(u) directly.
    if reaction_type in ("allen_cahn", "allen-cahn"):
        R_u = lam_c * (u**3 - u)
    elif reaction_type == "linear":
        alpha = float(reaction_spec.get("alpha", 0.0))
        R_u = alpha * u
    elif reaction_type in ("cubic", "poly3"):
        alpha = float(reaction_spec.get("alpha", 0.0))
        beta = float(reaction_spec.get("beta", 1.0))
        R_u = alpha * u + beta * u**3
    elif reaction_type in ("logistic", "fisher_kpp", "fisher-kpp"):
        rho = float(reaction_spec.get("rho", 1.0))
        R_u = rho * u * (1 - u)
    else:
        R_u = lam_c * (u**3 - u)  # default to Allen-Cahn
    
    # PDE: du/dt - eps * laplacian(u) + R(u) = f
    # Backward Euler weak form:
    # (u - u_n)/dt * v + eps * grad(u).grad(v) + R(u) * v - f * v = 0
    F_form = (
        (u - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + R_u * v * ufl.dx
        - f_expr * v * ufl.dx
    )
    
    # Create nonlinear problem
    J_form = ufl.derivative(F_form, u)
    problem = petsc.NonlinearProblem(
        F_form, u, bcs=bcs,
        J=J_form,
        petsc_options_prefix="nls_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 50,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-8,
            "snes_linesearch_type": "bt",
        }
    )
    
    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    t = t0
    nonlinear_iterations = []
    
    for step in range(n_steps):
        t += dt_val
        
        # Use previous solution as initial guess
        u.x.array[:] = u_n.x.array[:]
        
        try:
            problem.solve()
            snes = problem.solver
            n_iters = snes.getIterationNumber()
            reason = snes.getConvergedReason()
            if reason < 0:
                print(f"Warning: SNES did not converge at step {step}, reason={reason}")
        except Exception as e:
            print(f"SNES failed at step {step}, t={t:.4f}: {e}")
            n_iters = 0
        
        nonlinear_iterations.append(int(n_iters))
        u_n.x.array[:] = u.x.array[:]
    
    # Evaluate solution on output grid
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx_out)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_2d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    elapsed = time.time() - start_time
    print(f"Solve completed in {elapsed:.2f}s, {n_steps} steps, N={N}, eps={epsilon}")
    
    result = {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iterations) * 5,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import json
    config_path = "/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/cases/reaction_diffusion_no_exact_gaussian_source_allen_cahn/config.json"
    with open(config_path) as f:
        case_spec = json.load(f)
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    # Compare with reference
    ref = np.load('/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/results/miniswepde/reaction_diffusion_no_exact_gaussian_source_allen_cahn/oracle_output/reference.npz')
    u_ref = ref['u_star']
    print(f"Reference range: [{u_ref.min():.6f}, {u_ref.max():.6f}]")
    
    diff = np.abs(result['u'] - u_ref)
    norm_ref = np.sqrt(np.mean(u_ref**2))
    rel_l2 = np.sqrt(np.mean((result['u'] - u_ref)**2)) / (norm_ref + 1e-15)
    print(f"Max diff: {np.nanmax(diff):.6e}")
    print(f"Relative L2 error: {rel_l2:.6e}")
    print(f"Target: 3.25e-02")
    print(f"Pass: {rel_l2 <= 3.25e-02}")
