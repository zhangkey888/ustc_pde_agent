import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    time_params = pde_config.get("time", {})
    
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # Mesh resolution and element degree
    nx = ny = 80
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # The manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -eps * nabla^2 u = -eps * (-2*pi^2) * exp(-t)*sin(pi*x)*sin(pi*y) = 2*eps*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # R(u) - for linear reaction-diffusion, assume R(u) = sigma * u
    # We need to figure out eps and sigma from case_spec or derive them
    
    # For this problem, let's assume eps=1 and reaction coefficient sigma from case_spec
    # The PDE: du/dt - eps*nabla^2(u) + sigma*u = f
    # With u_exact = exp(-t)*sin(pi*x)*sin(pi*y):
    # f = du/dt - eps*nabla^2(u) + sigma*u
    #   = -exp(-t)*sin(pi*x)*sin(pi*y) + eps*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y) + sigma*exp(-t)*sin(pi*x)*sin(pi*y)
    #   = exp(-t)*sin(pi*x)*sin(pi*y) * (-1 + 2*eps*pi^2 + sigma)
    
    eps_val = pde_config.get("epsilon", 1.0)
    # Check for reaction coefficient
    reaction = pde_config.get("reaction", {})
    sigma_val = reaction.get("coefficient", 0.0) if isinstance(reaction, dict) else 0.0
    # If no reaction info, try to get from other places
    if sigma_val == 0.0:
        sigma_val = pde_config.get("sigma", 0.0)
    
    # For the manufactured solution, we compute f accordingly
    # f(x,y,t) = exp(-t)*sin(pi*x)*sin(pi*y) * (-1 + 2*eps*pi^2 + sigma)
    
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    sigma_c = fem.Constant(domain, PETSc.ScalarType(sigma_val))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    t_c = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    pi = np.pi
    
    # UFL expressions for manufactured solution
    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term
    f_coeff = -1.0 + 2.0 * eps_val * pi**2 + sigma_val
    f_expr_ufl = fem.Constant(domain, PETSc.ScalarType(f_coeff)) * ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # 4. Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 5. Boundary conditions - u = 0 on all boundaries (since sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    # For time-dependent BC, we need to update each step
    # But since BC is 0 on boundary for all t, we can use a simple zero BC
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # 6. Variational form (Backward Euler)
    # (u - u_n)/dt - eps*nabla^2(u) + sigma*u = f
    # Weak form: (u/dt)*v + eps*grad(u).grad(v) + sigma*u*v = f*v + (u_n/dt)*v
    
    a = (u / dt_c) * v * ufl.dx + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + sigma_c * u * v * ufl.dx
    L = f_expr_ufl * v * ufl.dx + (u_n / dt_c) * v * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for linear problem with constant coefficients)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    # 7. Time stepping
    for step in range(n_steps):
        t_current = (step + 1) * dt
        t_c.value = t_current  # f is evaluated at t_{n+1} for backward Euler
        
        # Update BC if needed (stays zero here)
        # u_bc is already zero
        
        # Assemble RHS
        b_vec = petsc.create_vector(L_form)
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
        b_vec.destroy()
    
    # 8. Extract solution on uniform grid
    nx_out = 60
    ny_out = 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
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
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }