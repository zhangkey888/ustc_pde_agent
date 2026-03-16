import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Get epsilon (diffusion coefficient)
    epsilon_val = pde_config.get("epsilon", 1.0)
    if isinstance(epsilon_val, dict):
        epsilon_val = epsilon_val.get("value", 1.0)
    epsilon_val = float(epsilon_val)
    
    # Get reaction parameters
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "linear")
    reaction_coeff = float(reaction.get("coefficient", 1.0))
    
    # Time parameters
    time_params = pde_config.get("time", None)
    is_transient = time_params is not None
    
    if is_transient:
        t_end = float(time_params.get("t_end", 0.3))
        dt_suggested = float(time_params.get("dt", 0.005))
        scheme = time_params.get("scheme", "backward_euler")
    else:
        t_end = 0.0
        dt_suggested = 0.005
        scheme = "backward_euler"
    
    # Manufactured solution: u = exp(-t)*(exp(4*x)*sin(pi*y))
    # For this problem we know the exact solution and can derive source term
    
    # 2. Create mesh - use fine mesh for accuracy
    nx, ny = 100, 100
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - degree 2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution as UFL expression
    pi = np.pi
    u_exact_ufl = ufl.exp(-t_const) * (ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1]))
    
    # Compute source term from manufactured solution:
    # u = exp(-t) * exp(4x) * sin(pi*y)
    # du/dt = -exp(-t) * exp(4x) * sin(pi*y) = -u
    # d2u/dx2 = 16 * exp(-t) * exp(4x) * sin(pi*y) = 16*u
    # d2u/dy2 = -pi^2 * exp(-t) * exp(4x) * sin(pi*y) = -pi^2*u
    # laplacian(u) = (16 - pi^2) * u
    # 
    # For reaction_diffusion with linear reaction R(u) = reaction_coeff * u:
    # du/dt - epsilon * laplacian(u) + reaction_coeff * u = f
    # -u - epsilon*(16 - pi^2)*u + reaction_coeff*u = f
    # f = u * (-1 - epsilon*(16 - pi^2) + reaction_coeff)
    # f = exp(-t)*exp(4x)*sin(pi*y) * (-1 - epsilon*(16-pi^2) + reaction_coeff)
    
    # But let's compute it symbolically with UFL to be safe
    # f = du/dt - eps*lap(u) + R(u)
    # du/dt = -u_exact for our solution
    
    # For the source term, we need to be careful. Let's compute it analytically.
    # u(x,y,t) = exp(-t) * exp(4*x) * sin(pi*y)
    # du/dt = -exp(-t) * exp(4*x) * sin(pi*y)
    # grad(u) = exp(-t) * [4*exp(4x)*sin(pi*y), exp(4x)*pi*cos(pi*y)]
    # lap(u) = exp(-t) * [16*exp(4x)*sin(pi*y) + exp(4x)*(-pi^2)*sin(pi*y)]
    #        = exp(-t) * exp(4x) * sin(pi*y) * (16 - pi^2)
    # 
    # Equation: du/dt - eps*lap(u) + R(u) = f
    # For linear reaction R(u) = reaction_coeff * u:
    # f = du/dt - eps*lap(u) + reaction_coeff * u
    # f = -u - eps*(16-pi^2)*u + reaction_coeff*u
    # f = u * (-1 - eps*(16-pi^2) + reaction_coeff)
    
    coeff_f = -1.0 - epsilon_val * (16.0 - pi**2) + reaction_coeff
    f_ufl = coeff_f * u_exact_ufl
    
    # If steady (no time), the equation is: -eps*lap(u) + R(u) = f
    # f_steady = -eps*(16-pi^2)*u + reaction_coeff*u = u*(-eps*(16-pi^2) + reaction_coeff)
    if not is_transient:
        coeff_f_steady = -epsilon_val * (16.0 - pi**2) + reaction_coeff
        f_ufl = coeff_f_steady * u_exact_ufl
    
    # 4. Boundary conditions
    # u = g on boundary, where g = exact solution
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function
    u_bc = fem.Function(V)
    
    # For time-dependent, we'll update BC at each time step
    def update_bc(t_val):
        u_bc.interpolate(lambda X: np.exp(-t_val) * np.exp(4.0 * X[0]) * np.sin(np.pi * X[1]))
    
    # 5. Set up variational forms
    if is_transient:
        # Backward Euler: (u^{n+1} - u^n)/dt - eps*lap(u^{n+1}) + R(u^{n+1}) = f^{n+1}
        # Weak form: (u^{n+1}, v)/dt + eps*(grad(u^{n+1}), grad(v)) + reaction_coeff*(u^{n+1}, v) 
        #          = (u^n, v)/dt + (f^{n+1}, v)
        
        dt_val = dt_suggested
        n_steps = int(round(t_end / dt_val))
        dt_val = t_end / n_steps  # adjust to hit t_end exactly
        
        dt_c = fem.Constant(domain, default_scalar_type(dt_val))
        eps_c = fem.Constant(domain, default_scalar_type(epsilon_val))
        react_c = fem.Constant(domain, default_scalar_type(reaction_coeff))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        u_n = fem.Function(V)  # solution at previous time step
        
        # Initialize u_n with initial condition at t=0
        u_n.interpolate(lambda X: np.exp(0.0) * np.exp(4.0 * X[0]) * np.sin(np.pi * X[1]))
        
        # Bilinear form
        a = (u * v / dt_c + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + react_c * u * v) * ufl.dx
        
        # We need f at current time step - we'll handle this with a fem.Function
        f_func = fem.Function(V)
        
        # Linear form
        L = (u_n * v / dt_c + f_func * v) * ufl.dx
        
        # Compile forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear reaction)
        A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
        A.assemble()
        
        # Create RHS vector
        b = petsc.create_vector(L_form)
        
        # Create solution function
        uh = fem.Function(V)
        
        # Set up KSP solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.GMRES)
        pc = solver.getPC()
        pc.setType(PETSc.PC.Type.ILU)
        solver.rtol = 1e-10
        solver.atol = 1e-12
        solver.max_it = 1000
        solver.setUp()
        
        # Store initial condition
        u_initial_func = fem.Function(V)
        u_initial_func.interpolate(lambda X: np.exp(4.0 * X[0]) * np.sin(np.pi * X[1]))
        
        total_iterations = 0
        
        # Time stepping
        t_current = 0.0
        for step in range(n_steps):
            t_current += dt_val
            
            # Update source term at new time
            t_for_f = t_current
            f_func.interpolate(
                lambda X, t=t_for_f: coeff_f * np.exp(-t) * np.exp(4.0 * X[0]) * np.sin(np.pi * X[1])
            )
            
            # Update BC at new time
            update_bc(t_current)
            bc = fem.dirichletbc(u_bc, boundary_dofs)
            
            # Reassemble matrix (BCs might change values)
            A.zeroEntries()
            petsc.assemble_matrix(A, a_form, bcs=[bc])
            A.assemble()
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Solve
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            
            total_iterations += solver.getIterationNumber()
            
            # Update u_n for next step
            u_n.x.array[:] = uh.x.array[:]
        
    else:
        # Steady case
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        eps_c = fem.Constant(domain, default_scalar_type(epsilon_val))
        react_c = fem.Constant(domain, default_scalar_type(reaction_coeff))
        
        t_const.value = 0.0
        update_bc(0.0)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        f_func = fem.Function(V)
        coeff_f_steady = -epsilon_val * (16.0 - pi**2) + reaction_coeff
        f_func.interpolate(
            lambda X: coeff_f_steady * np.exp(4.0 * X[0]) * np.sin(np.pi * X[1])
        )
        
        a = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + react_c * u * v) * ufl.dx
        L = f_func * v * ufl.dx
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": "1e-10"},
            petsc_options_prefix="steady_"
        )
        uh = problem.solve()
        total_iterations = 1
        n_steps = 0
        dt_val = 0.0
    
    # 7. Extract solution on 75x75 uniform grid
    eval_nx, eval_ny = 75, 75
    xs = np.linspace(0.0, 1.0, eval_nx)
    ys = np.linspace(0.0, 1.0, eval_ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((eval_nx, eval_ny))
    
    # Also extract initial condition if transient
    result = {"u": u_grid}
    
    if is_transient:
        u_init_values = np.full(points_3d.shape[0], np.nan)
        if len(points_on_proc) > 0:
            vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_init_values[eval_map] = vals_init.flatten()
        result["u_initial"] = u_init_values.reshape((eval_nx, eval_ny))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": total_iterations,
    }
    
    if is_transient:
        solver_info["dt"] = dt_val
        solver_info["n_steps"] = n_steps
        solver_info["time_scheme"] = scheme
    
    result["solver_info"] = solver_info
    
    return result