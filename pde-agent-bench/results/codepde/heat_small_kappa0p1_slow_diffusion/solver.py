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
    
    kappa = float(coeffs.get("kappa", 0.1))
    t_end = float(time_params.get("t_end", 0.2))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose parameters for accuracy
    N = 80  # mesh resolution
    degree = 2  # quadratic elements for better accuracy
    dt = 0.005  # smaller dt for accuracy with small kappa
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Manufactured solution: u = exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # du/dt = -0.5*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    # laplacian u = -(4*pi^2 + pi^2)*exp(-0.5*t)*sin(2*pi*x)*sin(pi*y) = -5*pi^2*u
    # f = du/dt - kappa * laplacian u = -0.5*u + kappa*5*pi^2*u = u*(-0.5 + 5*kappa*pi^2)
    # At time t: f = exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)*(-0.5 + 5*kappa*pi^2)
    
    # Time constant for source
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution as UFL expression (for BC and source)
    u_exact_ufl = ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term
    f_coeff = -0.5 + 5.0 * kappa * pi**2
    f_expr = f_coeff * ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # 4. Initial condition
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    # At t=0: u0 = sin(2*pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(2 * pi * X[0]) * np.sin(pi * X[1]))
    
    # Store initial condition for output
    # We'll evaluate on grid later
    
    # 5. Variational form (backward Euler)
    # (u^{n+1} - u^n)/dt - kappa * laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_expr * v) * ufl.dx
    
    # 6. Boundary conditions
    # u = g on boundary, where g = exact solution at current time
    # Since exact solution has sin(2*pi*x)*sin(pi*y), it's zero on all boundaries of [0,1]^2
    # sin(2*pi*0) = 0, sin(2*pi*1) = 0, sin(pi*0) = 0, sin(pi*1) = 0
    # So homogeneous Dirichlet BC
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 7. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Solution function
    u_sol = fem.Function(V, name="u_sol")
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 8. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
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
    
    # 9. Extract solution on 50x50 uniform grid
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
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on the same grid
    u_n_init = fem.Function(V)
    u_n_init.interpolate(lambda X: np.sin(2 * pi * X[0]) * np.sin(pi * X[1]))
    
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