import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry, nls
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.01)
    time_scheme = time_params.get("scheme", "backward_euler")
    
    # Diffusion coefficient
    epsilon = pde_config.get("epsilon", pde_config.get("diffusion_coefficient", 1.0))
    
    # Reaction type
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "linear")
    reaction_coeff = reaction.get("coefficient", 0.0)
    
    # Mesh and solver parameters
    nx = 60
    ny = 60
    degree = 2
    dt = dt_suggested
    
    # 2. Create mesh - quadrilateral as specified in case ID
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = exp(-t)*(exp(x)*sin(pi*y))
    pi = np.pi
    
    # Exact solution as UFL expression (for BCs and source term)
    u_exact_ufl = ufl.exp(-t_const) * (ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # Compute source term f from manufactured solution:
    # u_t = -exp(-t)*exp(x)*sin(pi*y)
    # u_xx = exp(-t)*exp(x)*sin(pi*y)
    # u_yy = -pi^2 * exp(-t)*exp(x)*sin(pi*y)
    # laplacian(u) = u_xx + u_yy = exp(-t)*exp(x)*sin(pi*y)*(1 - pi^2)
    # 
    # PDE: du/dt - eps*laplacian(u) + R(u) = f
    # du/dt = -u
    # -eps*laplacian(u) = -eps*(1-pi^2)*u = eps*(pi^2-1)*u
    # 
    # For linear reaction R(u) = reaction_coeff * u:
    # f = -u + eps*(pi^2 - 1)*u + reaction_coeff*u
    #   = u*(-1 + eps*(pi^2-1) + reaction_coeff)
    
    # But let's compute it symbolically to be safe with any reaction type
    # f = du/dt - eps*lap(u) + R(u)
    # du/dt = -exp(-t)*exp(x)*sin(pi*y) = -u_exact
    
    eps_const = fem.Constant(domain, PETSc.ScalarType(epsilon))
    
    # For the source term, we need the exact derivatives
    # u_exact = exp(-t)*exp(x[0])*sin(pi*x[1])
    # grad(u_exact) = exp(-t)*(exp(x[0])*sin(pi*x[1]), exp(x[0])*pi*cos(pi*x[1]))
    # lap(u_exact) = exp(-t)*(exp(x[0])*sin(pi*x[1]) - pi^2*exp(x[0])*sin(pi*x[1]))
    #              = exp(-t)*exp(x[0])*sin(pi*x[1])*(1 - pi^2)
    
    # du/dt = -exp(-t)*exp(x[0])*sin(pi*x[1]) = -u_exact
    
    # Source: f = du/dt - eps*lap(u) + R(u)
    u_exact_spatial = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # lap_u = (1 - pi^2) * u_exact
    lap_u_exact = (1.0 - ufl.pi**2) * u_exact_ufl
    
    # du/dt = -u_exact
    dudt_exact = -u_exact_ufl
    
    # Reaction term
    if reaction_type == "linear":
        R_u_exact = reaction_coeff * u_exact_ufl
    elif reaction_type == "quadratic" or reaction_type == "u_squared":
        R_u_exact = reaction_coeff * u_exact_ufl**2
    elif reaction_type == "cubic" or reaction_type == "u_cubed":
        R_u_exact = reaction_coeff * u_exact_ufl**3
    elif reaction_type == "fisher" or reaction_type == "logistic":
        R_u_exact = reaction_coeff * u_exact_ufl * (1.0 - u_exact_ufl)
    else:
        R_u_exact = reaction_coeff * u_exact_ufl
    
    f_expr = dudt_exact - eps_const * lap_u_exact + R_u_exact
    
    # 4. Set up the problem
    # For transient problem with backward Euler:
    # (u^{n+1} - u^n)/dt - eps*lap(u^{n+1}) + R(u^{n+1}) = f^{n+1}
    
    # If reaction is linear: R(u) = reaction_coeff * u, problem is linear at each time step
    # If nonlinear, we need Newton iteration
    
    is_nonlinear = reaction_type in ["quadratic", "u_squared", "cubic", "u_cubed", "fisher", "logistic"]
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Current solution and previous time step
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # current solution (for nonlinear) or trial (for linear)
    
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x,0) = exp(0)*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)
    t_const.value = 0.0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    u_h.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Store initial condition for output
    # Create grid for evaluation
    nx_out = 60
    ny_out = 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    def evaluate_on_grid(u_func):
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
        
        u_values = np.full(points_2d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            pts = np.array(points_on_proc)
            cls = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts, cls)
            u_values[eval_map] = vals.flatten()
        
        return u_values.reshape(nx_out, ny_out)
    
    # Evaluate initial condition
    u_initial_grid = evaluate_on_grid(u_n)
    
    # Boundary conditions - update at each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Time stepping
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    dt_const.value = dt
    
    total_linear_iterations = 0
    nonlinear_iterations_list = []
    
    if not is_nonlinear:
        # Linear reaction: use TrialFunction approach
        u_trial = ufl.TrialFunction(V)
        
        # Backward Euler: (u^{n+1} - u^n)/dt - eps*lap(u^{n+1}) + R(u^{n+1}) = f^{n+1}
        # Bilinear form (LHS):
        a_form = (u_trial * v / dt_const) * ufl.dx \
                 + eps_const * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx \
                 + reaction_coeff * u_trial * v * ufl.dx
        
        # Linear form (RHS):
        L_form = (u_n * v / dt_const) * ufl.dx + f_expr * v * ufl.dx
        
        # Compile forms
        a_compiled = fem.form(a_form)
        L_compiled = fem.form(L_form)
        
        # Assemble matrix (constant in time for linear reaction with constant eps)
        A = petsc.assemble_matrix(a_compiled, bcs=[bc])
        A.assemble()
        
        b = petsc.create_vector(L_compiled)
        
        # Set up solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.GMRES)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
        
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        
        for step in range(n_steps):
            t_current = (step + 1) * dt
            t_const.value = t_current
            
            # Update BC
            u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
            
            # Reassemble matrix (BCs may change pattern - safer to reassemble)
            A.zeroEntries()
            petsc.assemble_matrix(A, a_compiled, bcs=[bc])
            A.assemble()
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_compiled)
            petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            solver.solve(b, u_h.x.petsc_vec)
            u_h.x.scatter_forward()
            
            total_linear_iterations += solver.getIterationNumber()
            
            # Update u_n
            u_n.x.array[:] = u_h.x.array[:]
        
        solver.destroy()
        A.destroy()
        b.destroy()
        
    else:
        # Nonlinear reaction: use Newton solver
        # Residual form F = 0
        # (u_h - u_n)/dt * v + eps*grad(u_h)*grad(v) + R(u_h)*v - f*v = 0
        
        if reaction_type in ["quadratic", "u_squared"]:
            R_uh = reaction_coeff * u_h**2
        elif reaction_type in ["cubic", "u_cubed"]:
            R_uh = reaction_coeff * u_h**3
        elif reaction_type in ["fisher", "logistic"]:
            R_uh = reaction_coeff * u_h * (1.0 - u_h)
        else:
            R_uh = reaction_coeff * u_h
        
        F_form = ((u_h - u_n) / dt_const) * v * ufl.dx \
                 + eps_const * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
                 + R_uh * v * ufl.dx \
                 - f_expr * v * ufl.dx
        
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        
        for step in range(n_steps):
            t_current = (step + 1) * dt
            t_const.value = t_current
            
            # Update BC
            u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
            
            # Set initial guess from previous step
            u_h.x.array[:] = u_n.x.array[:]
            
            problem = petsc.NonlinearProblem(F_form, u_h, bcs=[bc])
            newton_solver = nls.petsc.NewtonSolver(domain.comm, problem)
            newton_solver.convergence_criterion = "incremental"
            newton_solver.rtol = 1e-10
            newton_solver.atol = 1e-12
            newton_solver.max_it = 50
            
            ksp = newton_solver.krylov_solver
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp_pc = ksp.getPC()
            ksp_pc.setType(PETSc.PC.Type.ILU)
            
            n_newton, converged = newton_solver.solve(u_h)
            assert converged, f"Newton solver did not converge at step {step+1}"
            u_h.x.scatter_forward()
            
            nonlinear_iterations_list.append(int(n_newton))
            
            # Update u_n
            u_n.x.array[:] = u_h.x.array[:]
    
    # 7. Extract solution on grid
    u_grid = evaluate_on_grid(u_h)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    if not is_nonlinear:
        solver_info["iterations"] = total_linear_iterations
    else:
        solver_info["nonlinear_iterations"] = nonlinear_iterations_list
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }
    
    return result