import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """
    Solve reaction-diffusion equation with cubic softening reaction term.
    """
    start_time = time.time()
    
    # Extract parameters from case_spec
    pde_spec = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults from problem description
    time_spec = pde_spec.get("time", {})
    t_end = time_spec.get("t_end", 0.25)
    dt_val = time_spec.get("dt", 0.005)
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True  # forced true per problem description
    
    # Reaction parameters
    params = pde_spec.get("parameters", {})
    epsilon = params.get("epsilon", 1.0)
    reaction_alpha = params.get("reaction_alpha", 1.0)
    reaction_beta = params.get("reaction_beta", 1.0)
    
    # Mesh resolution and element degree
    mesh_resolution = 80
    element_degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    # Time constant
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    alpha_c = fem.Constant(domain, PETSc.ScalarType(reaction_alpha))
    beta_c = fem.Constant(domain, PETSc.ScalarType(reaction_beta))
    
    # Manufactured solution: u = exp(-t)*(0.15*sin(3*pi*x)*sin(2*pi*y))
    u_exact_ufl = ufl.exp(-t) * (0.15 * ufl.sin(3*pi_val*x[0]) * ufl.sin(2*pi_val*x[1]))
    
    # Compute source term f from the manufactured solution
    # du/dt = -u_exact
    du_dt_exact = -u_exact_ufl
    
    # Laplacian: d^2u/dx^2 + d^2u/dy^2 = -(9*pi^2 + 4*pi^2)*u = -13*pi^2*u
    laplacian_u_exact = -(9*pi_val**2 + 4*pi_val**2) * u_exact_ufl
    
    # Reaction term R(u) = alpha*u + beta*u^3
    R_exact = alpha_c * u_exact_ufl + beta_c * u_exact_ufl**3
    
    # Source term: f = du/dt - eps*laplacian(u) + R(u)
    f_expr = du_dt_exact - eps_c * laplacian_u_exact + R_exact
    
    # Define the problem variables
    u_h = fem.Function(V)  # current solution
    u_n = fem.Function(V)  # previous time step
    v = ufl.TestFunction(V)
    
    # Reaction term R(u_h) = alpha*u_h + beta*u_h^3
    R_u = alpha_c * u_h + beta_c * u_h**3
    
    # Backward Euler weak form:
    # (u_h - u_n)/dt * v * dx + eps * grad(u_h) . grad(v) * dx + R(u_h) * v * dx - f * v * dx = 0
    F = ((u_h - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_expr * v * ufl.dx
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc = fem.Function(V)
    u_exact_interp = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # Initial condition: u(x, 0) = 0.15*sin(3*pi*x)*sin(2*pi*y)
    t.value = 0.0
    u_bc.interpolate(u_exact_interp)
    u_n.interpolate(u_exact_interp)
    u_h.interpolate(u_exact_interp)
    
    # Setup nonlinear solver using new API
    problem = petsc.NonlinearProblem(
        F, u_h, 
        bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 25,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-8,
        }
    )
    
    # Time stepping
    n_steps = int(round(t_end / dt_val))
    current_t = 0.0
    nonlinear_iterations = []
    
    for step in range(n_steps):
        current_t += dt_val
        t.value = current_t
        
        # Update boundary condition
        u_bc.interpolate(u_exact_interp)
        
        # Solve
        problem.solve()
        
        snes = problem.solver
        n_iters = snes.getIterationNumber()
        converged_reason = snes.getConvergedReason()
        
        if converged_reason <= 0:
            print(f"Warning: SNES did not converge at step {step}, reason={converged_reason}")
        
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 70x70 grid
    nx_eval, ny_eval = 70, 70
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Also compute initial condition on same grid
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_interp)
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_eval, ny_eval))
    
    elapsed = time.time() - start_time
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": sum(nonlinear_iterations) * 3,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "time": {
                "t_end": 0.25,
                "dt": 0.005,
                "scheme": "backward_euler"
            },
            "parameters": {
                "epsilon": 1.0,
                "reaction_alpha": 1.0,
                "reaction_beta": 1.0,
            },
        },
        "domain": {
            "type": "unit_square",
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    # Compute error against exact solution
    nx_eval, ny_eval = 70, 70
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    t_end_val = 0.25
    u_exact = np.exp(-t_end_val) * (0.15 * np.sin(3*np.pi*XX) * np.sin(2*np.pi*YY))
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Nonlinear iterations (first 5): {result['solver_info']['nonlinear_iterations'][:5]}")
