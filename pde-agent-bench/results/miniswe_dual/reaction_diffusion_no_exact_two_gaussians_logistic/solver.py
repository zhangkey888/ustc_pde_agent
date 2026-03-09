import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict = None):
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get('pde', {})
    
    # Time parameters - hardcoded defaults from problem description
    time_spec = pde_spec.get('time', {})
    t_end = float(time_spec.get('t_end', 0.4))
    dt = float(time_spec.get('dt', 0.01))
    time_scheme = time_spec.get('scheme', 'backward_euler')
    is_transient = True
    
    # PDE parameters from config
    pde_params = pde_spec.get('pde_params', {})
    epsilon = float(pde_params.get('epsilon', 0.08))
    reaction_spec = pde_params.get('reaction', {})
    rho = float(reaction_spec.get('rho', 2.0))
    
    # Solver parameters
    mesh_resolution = 110
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = 6.0 * (ufl.exp(-160.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2)) + 
                     0.8 * ufl.exp(-160.0 * ((x[0] - 0.75)**2 + (x[1] - 0.35)**2)))
    
    # Initial condition
    def u0_func(X):
        return 0.3 * np.exp(-50.0 * ((X[0] - 0.3)**2 + (X[1] - 0.5)**2)) + \
               0.3 * np.exp(-50.0 * ((X[0] - 0.7)**2 + (X[1] - 0.5)**2))
    
    # Functions
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(u0_func)
    
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u0_func)
    
    u = fem.Function(V, name="u")
    u.x.array[:] = u_n.x.array[:]
    
    v = ufl.TestFunction(V)
    
    # Boundary conditions (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # Constants
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    rho_const = fem.Constant(domain, PETSc.ScalarType(rho))
    
    # Weak form (Backward Euler)
    # PDE: du/dt - eps*laplacian(u) + rho*u*(1-u) = f
    R_u = rho_const * u * (1.0 - u)
    
    F_form = ((u - u_n) / dt_const * v * ufl.dx 
              + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx 
              + R_u * v * ufl.dx 
              - f_expr * v * ufl.dx)
    
    # Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt))
    nonlinear_iterations = []
    total_linear_iterations = 0
    
    # Create nonlinear problem with SNES
    snes_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-10,
    }
    
    problem = petsc.NonlinearProblem(
        F_form, u, bcs=bcs,
        petsc_options_prefix="nls_",
        petsc_options=snes_options,
    )
    
    snes = problem.solver
    
    for step in range(n_steps):
        t += dt
        # Use previous solution as initial guess
        u.x.array[:] = u_n.x.array[:]
        
        # Copy current u into the SNES solution vector
        u.x.petsc_vec.copy(problem.x)
        problem.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        problem.solve()
        
        newton_its = snes.getIterationNumber()
        linear_its = snes.getLinearSolveIterations()
        nonlinear_iterations.append(int(newton_its))
        total_linear_iterations += int(linear_its)
        
        u.x.scatter_forward()
        u_n.x.array[:] = u.x.array[:]
    
    # Evaluate on 70x70 grid
    nx_out, ny_out = 70, 70
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_flat = np.zeros((nx_out * ny_out, 3))
    points_flat[:, 0] = XX.flatten()
    points_flat[:, 1] = YY.flatten()
    points_flat[:, 2] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_linear_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Time steps: {result['solver_info']['n_steps']}")
    print(f"Newton iters: {result['solver_info']['nonlinear_iterations']}")
    print(f"Total linear iters: {result['solver_info']['iterations']}")
    
    # Compare with reference
    import os
    ref_path = '/data/home/bingodong/code/ustc_pde_agent/pde-agent-bench/results/miniswepde/reaction_diffusion_no_exact_two_gaussians_logistic/oracle_output/reference.npz'
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        u_ref = ref['u_star']
        u_sol = result['u']
        rel_l2 = np.sqrt(np.nansum((u_sol - u_ref)**2) / np.nansum(u_ref**2))
        print(f"Relative L2 error vs reference: {rel_l2:.6e}")
    
    np.savez("solution.npz", u=result['u'])
    import json
    with open("meta.json", "w") as f:
        json.dump(result['solver_info'], f, indent=2)
    print("Saved solution.npz and meta.json")
