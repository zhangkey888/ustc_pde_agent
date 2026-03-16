import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    kappa = 1.0
    coeffs = pde_config.get("coefficients", {})
    if "kappa" in coeffs:
        kappa = float(coeffs["kappa"])
    
    time_params = pde_config.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use finer dt for accuracy
    dt = 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # 2. Create mesh - use fine mesh for accuracy
    nx, ny = 100, 100
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinates and exact solution
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Manufactured solution: u = exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y))
    # We need a time parameter
    t_param = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    u_exact_ufl = ufl.exp(-t_param) * ufl.sin(3 * pi * (x[0] + x[1])) * ufl.sin(pi * (x[0] - x[1]))
    
    # Compute source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y)) = -u
    # We need laplacian of the spatial part:
    # u_spatial = sin(3*pi*(x+y))*sin(pi*(x-y))
    # Let a = x+y, b = x-y
    # u_spatial = sin(3*pi*a)*sin(pi*b)
    # d/dx = d/da + d/db, d/dy = d/da - d/db
    # d^2/dx^2 = d^2/da^2 + 2*d^2/(da db) + d^2/db^2
    # d^2/dy^2 = d^2/da^2 - 2*d^2/(da db) + d^2/db^2
    # laplacian = 2*d^2/da^2 + 2*d^2/db^2
    # d^2/da^2 [sin(3*pi*a)*sin(pi*b)] = -9*pi^2*sin(3*pi*a)*sin(pi*b)
    # d^2/db^2 [sin(3*pi*a)*sin(pi*b)] = -pi^2*sin(3*pi*a)*sin(pi*b)
    # laplacian = 2*(-9*pi^2 - pi^2)*sin(3*pi*a)*sin(pi*b) = -20*pi^2 * u_spatial
    # So laplacian(u) = -20*pi^2 * u
    # f = du/dt - kappa*laplacian(u) = -u - kappa*(-20*pi^2)*u = (-1 + 20*kappa*pi^2)*u
    
    f_ufl = (-1.0 + 20.0 * kappa * pi**2) * u_exact_ufl
    
    # 5. Define variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - kappa*laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initialize u_n with exact solution at t=0
    u_n.interpolate(lambda x_arr: np.exp(0.0) * np.sin(3 * pi * (x_arr[0] + x_arr[1])) * np.sin(pi * (x_arr[0] - x_arr[1])))
    
    # Store initial condition for output
    # Create evaluation grid first
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Bilinear form (LHS)
    a_form = (u_trial * v_test / dt_const + kappa_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test))) * ufl.dx
    
    # Linear form (RHS)
    L_form = (u_n * v_test / dt_const + f_ufl * v_test) * ufl.dx
    
    # 6. Boundary conditions - Dirichlet from exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Set initial BC values at t=0
    # We'll update these each time step
    bc = fem.dirichletbc(bc_func, bc_dofs)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant since kappa and dt don't change)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_compiled)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Evaluate initial condition on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    u_initial_flat = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc_arr, cells_on_proc_arr)
        u_initial_flat[eval_map] = vals.flatten()
    u_initial = u_initial_flat.reshape(nx_eval, ny_eval)
    
    # 7. Time stepping
    total_iterations = 0
    t_current = 0.0
    
    for step in range(n_steps):
        t_current += dt
        
        # Update time parameter for source term and BC
        t_param.value = t_current
        
        # Update BC function
        bc_func.interpolate(
            lambda x_arr, t=t_current: np.exp(-t) * np.sin(3 * pi * (x_arr[0] + x_arr[1])) * np.sin(pi * (x_arr[0] - x_arr[1]))
        )
        
        # Reassemble matrix (BCs might change pattern - but actually A is constant, just need to re-apply BCs)
        # Since A doesn't change (same kappa, dt), we can reuse it
        # But BC values change, so we need to handle that in the RHS
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        
        # Apply lifting for non-zero Dirichlet BCs
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on evaluation grid
    u_final_flat = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc_arr, cells_on_proc_arr)
        u_final_flat[eval_map] = vals.flatten()
    u_grid = u_final_flat.reshape(nx_eval, ny_eval)
    
    # Clean up PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }