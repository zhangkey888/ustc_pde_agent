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
    coeffs = pde_config.get("coefficients", {})
    
    kappa = float(coeffs.get("kappa", 1.0))
    t_end = float(time_config.get("t_end", 0.08))
    dt_suggested = float(time_config.get("dt", 0.008))
    scheme = time_config.get("scheme", "backward_euler")
    
    # Choose parameters for accuracy
    nx, ny = 80, 80
    degree = 2
    dt = 0.002  # smaller dt for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(2*pi*y)
    # -kappa * laplacian(u) = kappa * (pi^2 + 4*pi^2) * exp(-t)*sin(pi*x)*sin(2*pi*y)
    #                       = 5*kappa*pi^2 * exp(-t)*sin(pi*x)*sin(2*pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, equation is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt + kappa*(pi^2 + 4*pi^2)*u = -u + 5*kappa*pi^2*u = u*(-1 + 5*kappa*pi^2)
    
    # Time as a Constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # We'll use backward Euler: (u^{n+1} - u^n)/dt - kappa*laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Source term as UFL expression (depends on t_const)
    pi_ = np.pi
    # f = exp(-t) * sin(pi*x) * sin(2*pi*y) * (-1 + 5*kappa*pi^2)
    f_coeff = -1.0 + 5.0 * kappa * pi_**2
    f_expr = f_coeff * ufl.exp(-t_const) * ufl.sin(pi_ * x[0]) * ufl.sin(2.0 * pi_ * x[1])
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    kappa_c = fem.Constant(domain, default_scalar_type(kappa))
    
    # Bilinear form
    a = ufl.inner(u, v) * ufl.dx + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Linear form
    L = ufl.inner(u_n, v) * ufl.dx + dt_c * ufl.inner(f_expr, v) * ufl.dx
    
    # 4. Boundary conditions - exact solution on boundary
    # g = exp(-t)*sin(pi*x)*sin(2*pi*y)
    # On the unit square boundary, sin(pi*x)*sin(2*pi*y) = 0 on all edges:
    # x=0: sin(0)=0, x=1: sin(pi)=0, y=0: sin(0)=0, y=1: sin(2*pi)=0
    # So g = 0 on all boundaries for all time
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # 5. Initial condition: u(x, 0) = sin(pi*x)*sin(2*pi*y)
    u_n.interpolate(lambda x: np.sin(pi_ * x[0]) * np.sin(2.0 * pi_ * x[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 6. Compile forms and set up manual assembly for time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    # 7. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Assemble RHS
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
    
    # 8. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up
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