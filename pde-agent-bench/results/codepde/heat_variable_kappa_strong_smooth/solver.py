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
    dt = 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # Mesh resolution and element degree
    nx = ny = 80
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Manufactured solution: u = exp(-t)*sin(3*pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # kappa = 1 + 0.8*sin(2*pi*x)*sin(2*pi*y)
    kappa = 1.0 + 0.8 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(3*pi*x)*sin(2*pi*y)
    # We need to compute div(kappa * grad(u_exact)) symbolically
    # Let's define u_exact as a UFL expression and compute f
    dudt = -ufl.exp(-t_const) * ufl.sin(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f_expr = dudt + ufl.div(kappa * grad_u_exact)
    # Note: the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - div(kappa*grad(u_exact))  but with a minus sign:
    # Actually: du/dt - div(kappa*grad(u)) = f
    # => f = dudt - div(kappa*grad(u_exact))
    # But div(kappa*grad(u)) with a minus sign in the PDE means:
    # f = dudt + (-div(kappa*grad(u_exact)))... let me be careful.
    # PDE: du/dt - div(kappa * grad(u)) = f
    # For manufactured solution: f = du_exact/dt - div(kappa * grad(u_exact))
    f_source = dudt - ufl.div(kappa * grad_u_exact)
    
    # 4. Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(3*pi*x)*sin(2*pi*y)
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Save initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])
    
    # Build evaluation infrastructure once
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.array([], dtype=np.int32)
    
    def eval_on_grid(func):
        u_values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc, cells_on_proc)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)
    
    u_initial_grid = eval_on_grid(u_n)
    
    # 5. Backward Euler time stepping
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    # => (1/dt) * u * v dx + kappa * grad(u) . grad(v) dx = (1/dt) * u_n * v dx + f * v dx
    
    # Bilinear form (constant in time since kappa doesn't depend on t)
    a_form = (u * v / dt_const) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Linear form (changes each time step due to f and u_n)
    L_form = (u_n * v / dt_const) * ufl.dx + f_source * v * ufl.dx
    
    # 6. Boundary conditions - update each time step
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (kappa doesn't change, dt doesn't change, so A is constant)
    A = petsc.assemble_matrix(a_compiled, bcs=[])
    # We'll handle BCs properly - first let's set them up
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    # Set BC for t = dt (first step)
    t_const.value = dt
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # Reassemble A with BCs
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping loop
    for step in range(n_steps):
        t_new = (step + 1) * dt
        t_const.value = t_new
        
        # Update BC
        u_bc.interpolate(bc_expr)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        
        # Apply lifting and BCs
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 7. Extract solution on grid
    u_grid = eval_on_grid(u_sol)
    
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
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }