import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde_config.get("epsilon", 1.0)
    reaction_type = pde_config.get("reaction", {}).get("type", "linear")
    reaction_coeff = pde_config.get("reaction", {}).get("coefficient", 1.0)
    
    time_params = pde_config.get("time", None)
    
    # Manufactured solution: u = exp(-t)*(sin(4*pi*x)*sin(3*pi*y))
    # For this problem, we need to derive f from the PDE
    
    # Parameters
    nx_mesh = 100
    ny_mesh = 100
    degree = 2
    
    t_end = 0.3
    dt_val = 0.005
    scheme = "crank_nicolson"
    
    if time_params is not None:
        t_end = time_params.get("t_end", 0.3)
        dt_val = time_params.get("dt", 0.005)
        scheme = time_params.get("scheme", "crank_nicolson")
    
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_c = fem.Constant(domain, default_scalar_type(dt_val))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Compute source term f from: du/dt - eps * laplacian(u) + R(u) = f
    # u = exp(-t)*sin(4*pi*x)*sin(3*pi*y)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(3*pi*y) = -u
    # laplacian(u) = exp(-t)*[-(4*pi)^2 - (3*pi)^2]*sin(4*pi*x)*sin(3*pi*y)
    #              = -(16*pi^2 + 9*pi^2)*u = -25*pi^2*u
    # -eps*laplacian(u) = eps*25*pi^2*u
    # For linear reaction R(u) = reaction_coeff * u
    # f = du/dt - eps*laplacian(u) + R(u)
    #   = -u + eps*25*pi^2*u + reaction_coeff*u
    #   = u*(-1 + 25*eps*pi^2 + reaction_coeff)
    
    # But let's compute it symbolically to be safe
    # du/dt
    dudt_ufl = -ufl.exp(-t_const) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # -eps * laplacian(u) (note: -div(grad(u)) applied to the exact solution)
    # laplacian of sin(4*pi*x)*sin(3*pi*y) = -(16*pi^2 + 9*pi^2)*sin(4*pi*x)*sin(3*pi*y)
    # so -eps*laplacian(u) = eps*(25*pi^2)*u_exact
    neg_eps_laplacian = epsilon * (25.0 * pi**2) * u_exact_ufl
    
    # Reaction term
    if reaction_type == "linear":
        R_exact = reaction_coeff * u_exact_ufl
    elif reaction_type == "cubic" or reaction_type == "u_cubed":
        R_exact = reaction_coeff * u_exact_ufl**3
    else:
        R_exact = reaction_coeff * u_exact_ufl
    
    f_ufl = dudt_ufl + neg_eps_laplacian + R_exact
    
    # Define trial and test functions
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # Solution functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    # Set initial condition: u(x,0) = sin(4*pi*x)*sin(3*pi*y)
    t_const.value = 0.0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    # Store initial condition for output
    # Create evaluation grid
    nx_out = 70
    ny_out = 70
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])
    
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
    
    u_initial = evaluate_on_grid(u_n)
    
    # Boundary conditions - update each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # For Crank-Nicolson: 
    # (u^{n+1} - u^n)/dt - eps * 0.5*(laplacian(u^{n+1}) + laplacian(u^n)) 
    #   + 0.5*(R(u^{n+1}) + R(u^n)) = 0.5*(f^{n+1} + f^n)
    # For linear reaction R(u) = c*u, this is linear in u^{n+1}
    
    # theta = 0.5 for Crank-Nicolson, 1.0 for backward Euler
    if scheme == "crank_nicolson":
        theta = 0.5
    elif scheme == "backward_euler":
        theta = 1.0
    else:
        theta = 0.5
    
    theta_c = fem.Constant(domain, default_scalar_type(theta))
    one_minus_theta = fem.Constant(domain, default_scalar_type(1.0 - theta))
    eps_c = fem.Constant(domain, default_scalar_type(epsilon))
    
    # For linear reaction:
    # (u^{n+1} - u^n)/dt + eps*(-laplacian)(theta*u^{n+1} + (1-theta)*u^n) 
    #   + c*(theta*u^{n+1} + (1-theta)*u^n) = theta*f^{n+1} + (1-theta)*f^n
    
    # Bilinear form (LHS) - terms involving u^{n+1}
    # u^{n+1}/dt + theta*eps*grad(u^{n+1}).grad(v) + theta*c*u^{n+1}*v
    
    rc = fem.Constant(domain, default_scalar_type(reaction_coeff))
    
    if reaction_type == "linear":
        a_form = (u_trial * v_test / dt_c) * ufl.dx \
                + theta_c * eps_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx \
                + theta_c * rc * u_trial * v_test * ufl.dx
    else:
        # For nonlinear reaction, we'd need Newton. For now assume linear.
        a_form = (u_trial * v_test / dt_c) * ufl.dx \
                + theta_c * eps_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx \
                + theta_c * rc * u_trial * v_test * ufl.dx
    
    # We need f at time t^{n+1} and t^n
    # f depends on t_const, so we'll create two constants for the two times
    t_new = fem.Constant(domain, default_scalar_type(0.0))
    t_old = fem.Constant(domain, default_scalar_type(0.0))
    
    # f at t_new
    u_exact_new = ufl.exp(-t_new) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    dudt_new = -u_exact_new
    neg_eps_lap_new = epsilon * (25.0 * pi**2) * u_exact_new
    R_exact_new = reaction_coeff * u_exact_new
    f_new = dudt_new + neg_eps_lap_new + R_exact_new
    
    # f at t_old
    u_exact_old = ufl.exp(-t_old) * ufl.sin(4 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    dudt_old = -u_exact_old
    neg_eps_lap_old = epsilon * (25.0 * pi**2) * u_exact_old
    R_exact_old = reaction_coeff * u_exact_old
    f_old = dudt_old + neg_eps_lap_old + R_exact_old
    
    # RHS
    L_form = (u_n * v_test / dt_c) * ufl.dx \
            - one_minus_theta * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v_test)) * ufl.dx \
            - one_minus_theta * rc * u_n * v_test * ufl.dx \
            + (theta_c * f_new + one_minus_theta * f_old) * v_test * ufl.dx
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=[])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    # BC function
    u_bc_func = fem.Function(V)
    
    total_iterations = 0
    
    # Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt_val
        t_new.value = t_current
        t_old.value = t_current - dt_val
        
        # Update BC
        t_const.value = t_current
        u_bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc_func, boundary_dofs)
        
        # Re-assemble matrix with BCs (since BCs change each step)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()
        solver.setOperators(A)
        
        # Assemble RHS
        b_vec = petsc.create_vector(L_compiled)
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_compiled)
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_h.x.array[:]
        
        b_vec.destroy()
    
    # Evaluate on grid
    u_grid = evaluate_on_grid(u_h)
    
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }