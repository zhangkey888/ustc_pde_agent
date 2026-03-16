import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    
    # Use a smaller dt for accuracy
    dt_val = 0.005
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # exact division
    
    # Mesh resolution - need good accuracy for the localized kappa peak
    nx, ny = 80, 80
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt_val))
    
    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # kappa = 1 + 30*exp(-150*((x-0.35)**2 + (y-0.65)**2))
    kappa_ufl = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(pi*x)*sin(2*pi*y)
    dudt_ufl = -ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # div(kappa * grad(u)) needs to be computed
    # grad(u_exact) = exp(-t) * (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # We use UFL to compute this symbolically
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa_ufl * grad_u_exact)
    
    f_ufl = dudt_ufl - div_kappa_grad_u
    
    # 5. Define variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + (kappa * grad(u^{n+1}), grad(v)) = (f^{n+1}, v) + (u^n, v)/dt
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Bilinear form (LHS)
    a_form = (u_trial * v_test / dt_const) * ufl.dx + \
             ufl.inner(kappa_ufl * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    # Linear form (RHS)
    L_form = (u_n * v_test / dt_const) * ufl.dx + f_ufl * v_test * ufl.dx
    
    # 6. Boundary conditions - Dirichlet from exact solution
    # u_exact on boundary at current time
    u_bc_func = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # 7. Set initial condition: u(x, 0) = sin(pi*x)*sin(2*pi*y)
    u_exact_expr_t0 = fem.Expression(
        ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_exact_expr_t0)
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # 8. Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # 9. Assemble matrix (kappa doesn't depend on time, and dt is constant, so A is constant)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Expression for exact solution at current time (for BC updates)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    total_iterations = 0
    
    # 10. Time-stepping loop
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt_val
        t_const.value = current_t
        
        # Update boundary condition
        u_bc_func.interpolate(u_exact_expr)
        
        # Assemble RHS
        b_vec = petsc.create_vector(L_compiled)
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        
        # Apply lifting for Dirichlet BCs
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
        b_vec.destroy()
    
    # 11. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on grid
    # Reset time to 0 for initial condition evaluation
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr_t0)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }