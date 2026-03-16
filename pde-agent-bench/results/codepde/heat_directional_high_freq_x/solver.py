import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    coeffs = pde_config.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.08)
    dt_suggested = time_params.get("dt", 0.004)
    scheme = time_params.get("scheme", "backward_euler")
    
    # High frequency in x (8*pi), need fine mesh
    # u = exp(-t)*sin(8*pi*x)*sin(pi*y)
    # Need at least ~16 elements per wavelength in x direction
    # wavelength in x = 2/(8) = 0.25, so need ~64 elements minimum, use more for accuracy
    
    nx = 128
    ny = 32
    degree = 2
    dt = dt_suggested / 2.0  # Use smaller dt for accuracy: 0.002
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    # Manufactured solution: u_exact = exp(-t)*sin(8*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(8 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(8*pi*x)*sin(pi*y)
    # laplacian(u) = exp(-t)*[-(8*pi)^2 - pi^2]*sin(8*pi*x)*sin(pi*y)
    #              = -exp(-t)*(64*pi^2 + pi^2)*sin(8*pi*x)*sin(pi*y)
    #              = -exp(-t)*65*pi^2*sin(8*pi*x)*sin(pi*y)
    # f = du/dt - kappa * laplacian(u)
    #   = -exp(-t)*sin(8*pi*x)*sin(pi*y) - kappa*(-exp(-t)*65*pi^2*sin(8*pi*x)*sin(pi*y))  ... wait, sign
    # Actually: PDE is du/dt - div(kappa*grad(u)) = f
    # div(kappa*grad(u)) = kappa * laplacian(u) = kappa * (-65*pi^2) * exp(-t)*sin(8*pi*x)*sin(pi*y)
    # So f = du/dt - kappa*laplacian(u)
    #      = -exp(-t)*sin(...) - kappa*(-65*pi^2)*exp(-t)*sin(...)
    #      = exp(-t)*sin(8*pi*x)*sin(pi*y) * (-1 + 65*kappa*pi^2)
    
    f_ufl = ufl.exp(-t_const) * ufl.sin(8 * pi * x[0]) * ufl.sin(pi * x[1]) * (-1.0 + 65.0 * kappa * pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Set initial condition: u(x,0) = sin(8*pi*x)*sin(pi*y)
    t_const.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])
    
    # Evaluate initial condition
    def evaluate_function(u_func, pts):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(pts.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(pts.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full(pts.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values
    
    u_initial_vals = evaluate_function(u_n, points_2d)
    u_initial_grid = u_initial_vals.reshape((nx_out, ny_out))
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem with constant kappa and dt)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_exact_expr_bc = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_exact_expr_bc)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Extract solution on grid
    u_final_vals = evaluate_function(u_sol, points_2d)
    u_grid = u_final_vals.reshape((nx_out, ny_out))
    
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