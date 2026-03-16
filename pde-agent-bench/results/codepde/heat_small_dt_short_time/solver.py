import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    time_params = pde.get("time", {})
    
    kappa = float(coeffs.get("kappa", 1.0))
    t_end = float(time_params.get("t_end", 0.06))
    dt_suggested = float(time_params.get("dt", 0.003))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose parameters for accuracy
    N = 80  # mesh resolution
    degree = 2  # quadratic elements for better accuracy
    dt = dt_suggested  # use suggested dt
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust dt to exactly hit t_end
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # -kappa * laplacian(u) = kappa * exp(-t) * 32*pi^2 * sin(4*pi*x)*sin(4*pi*y)
    # f = du/dt - kappa*laplacian(u) (note: equation is du/dt - div(kappa*grad(u)) = f)
    # f = -exp(-t)*sin(4*pi*x)*sin(4*pi*y) + kappa*32*pi^2*exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # f = exp(-t)*sin(4*pi*x)*sin(4*pi*y)*(-1 + 32*kappa*pi^2)
    
    # Time as a Constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Source term (using UFL expressions with t_const)
    pi_val = np.pi
    sin4pix = ufl.sin(4 * pi_val * x[0])
    sin4piy = ufl.sin(4 * pi_val * x[1])
    
    # For backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    
    # f at time t_{n+1}
    f_expr = ufl.exp(-t_const) * sin4pix * sin4piy * (-1.0 + 32.0 * kappa * pi_val**2)
    
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    
    a_form = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L_form = (u_n * v / dt_c + f_expr * v) * ufl.dx
    
    # 4. Boundary conditions - exact solution on boundary
    # g = exp(-t)*sin(4*pi*x)*sin(4*pi*y) which is 0 on all boundaries of unit square
    # since sin(0) = sin(4*pi) = 0
    # So homogeneous Dirichlet BC
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value function - on the boundary sin(4*pi*x)*sin(4*pi*y) = 0 for unit square
    # But to be safe with numerical precision, use the exact solution
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 5. Set initial condition: u(x,0) = sin(4*pi*x)*sin(4*pi*y)
    u_n.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))
    
    # Store initial condition for output
    # Create evaluation grid first
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Evaluate initial condition on grid
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
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    u_initial_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc_arr, cells_on_proc_arr)
        u_initial_values[eval_map] = vals.flatten()
    u_initial_grid = u_initial_values.reshape((nx_out, ny_out))
    
    # 6. Compile forms and set up manual assembly for time stepping
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create solution function
    u_sol = fem.Function(V)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Create RHS vector
    b = petsc.create_vector(L_compiled)
    
    total_iterations = 0
    
    # 7. Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update BC (zero on boundary for this problem, but update for safety)
        # On unit square boundary, sin(4*pi*x)*sin(4*pi*y) = 0
        # so u_bc stays zero
        u_bc.x.array[:] = 0.0
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc_arr, cells_on_proc_arr)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Clean up
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
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