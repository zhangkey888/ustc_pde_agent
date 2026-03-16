import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    time_config = pde_config.get("time", {})
    coefficients = pde_config.get("coefficients", {})
    
    kappa = float(coefficients.get("kappa", 1.0))
    t_end = float(time_config.get("t_end", 0.08))
    dt_suggested = float(time_config.get("dt", 0.008))
    scheme = time_config.get("scheme", "backward_euler")
    
    # Use finer mesh and smaller dt for accuracy
    # The solution has exp(5*x) which creates a boundary layer near x=1
    # exp(5) ≈ 148.4, so we need good resolution
    nx, ny = 120, 50
    degree = 2
    
    # Use smaller dt for accuracy
    dt = dt_suggested / 2.0  # 0.004
    n_steps = int(np.round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-t)*exp(5*x)*sin(pi*y)
    # du/dt = -exp(-t)*exp(5*x)*sin(pi*y)
    # grad(u) = exp(-t)*[5*exp(5*x)*sin(pi*y), exp(5*x)*pi*cos(pi*y)]
    # laplacian(u) = exp(-t)*(25*exp(5*x)*sin(pi*y) - pi^2*exp(5*x)*sin(pi*y))
    #              = exp(-t)*exp(5*x)*sin(pi*y)*(25 - pi^2)
    # f = du/dt - kappa*laplacian(u)
    #   = -exp(-t)*exp(5*x)*sin(pi*y) - kappa*exp(-t)*exp(5*x)*sin(pi*y)*(25 - pi^2)
    #   = exp(-t)*exp(5*x)*sin(pi*y)*(-1 - kappa*(25 - pi^2))
    
    t_param = fem.Constant(domain, default_scalar_type(0.0))
    pi = np.pi
    
    # Source term as UFL expression
    f_coeff = -1.0 - kappa * (25.0 - pi**2)
    f_expr = ufl.exp(-t_param) * ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1]) * f_coeff
    
    # Exact solution UFL expression for BCs
    u_exact_ufl = ufl.exp(-t_param) * ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # 5. Set up time-stepping (backward Euler)
    # u_n is solution at previous time step
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")  # solution at current time step
    
    # Initial condition: u(x,0) = exp(0)*exp(5*x)*sin(pi*y) = exp(5*x)*sin(pi*y)
    u_n.interpolate(fem.Expression(
        ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    
    # Variational form for backward Euler:
    # (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_const) * ufl.dx + f_expr * v * ufl.dx
    
    # 6. Boundary conditions - all boundaries are Dirichlet
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Set up solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time-stepping loop
    for step in range(n_steps):
        t_current = (step + 1) * dt
        t_param.value = t_current
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on grid for analysis
    u_n_init = fem.Function(V)
    u_n_init.interpolate(fem.Expression(
        ufl.exp(5.0 * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n_init.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
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