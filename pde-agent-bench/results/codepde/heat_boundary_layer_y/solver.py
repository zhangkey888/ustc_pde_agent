import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa = float(coefficients.get("kappa", 1.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.08))
    dt_suggested = float(time_params.get("dt", 0.008))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy given boundary layer in y
    # The exact solution is u = exp(-t)*exp(5*y)*sin(pi*x)
    # exp(5*y) creates a boundary layer near y=1, need fine mesh there
    
    # Choose mesh resolution - need fine mesh especially in y direction for exp(5*y)
    nx = 80
    ny = 80
    degree = 2
    dt = dt_suggested / 2.0  # Use finer time step
    n_steps = int(np.round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(-t)*exp(5*y)*sin(pi*x)
    # du/dt = -exp(-t)*exp(5*y)*sin(pi*x)
    # laplacian u = exp(-t)*(-pi^2*sin(pi*x)*exp(5*y) + 25*exp(5*y)*sin(pi*x))
    #            = exp(-t)*exp(5*y)*sin(pi*x)*(25 - pi^2)
    # -kappa * laplacian u = -kappa * exp(-t)*exp(5*y)*sin(pi*x)*(25 - pi^2)
    # f = du/dt - kappa * laplacian u  (note: equation is du/dt - div(kappa grad u) = f)
    # Wait, the equation is: du/dt - div(kappa * grad(u)) = f
    # div(kappa * grad(u)) = kappa * laplacian(u) = kappa * exp(-t)*exp(5*y)*sin(pi*x)*(25 - pi^2)
    # f = du/dt - kappa * laplacian(u) = -exp(-t)*exp(5*y)*sin(pi*x) - kappa*(25 - pi^2)*exp(-t)*exp(5*y)*sin(pi*x)
    # f = exp(-t)*exp(5*y)*sin(pi*x) * (-1 - kappa*(25 - pi^2))
    
    pi_val = np.pi
    
    # Time parameter as a Constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Source term as UFL expression
    # f = exp(-t)*exp(5*x[1])*sin(pi*x[0]) * (-1 - kappa*(25 - pi^2))
    coeff_f = -1.0 - kappa * (25.0 - pi_val**2)
    f_expr = ufl.exp(-t_const) * ufl.exp(5.0 * x[1]) * ufl.sin(pi_val * x[0]) * coeff_f
    
    # Exact solution UFL expression for BCs
    u_exact_expr = ufl.exp(-t_const) * ufl.exp(5.0 * x[1]) * ufl.sin(pi_val * x[0])
    
    # 4. Define functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    u_bc_func = fem.Function(V)  # boundary condition function
    
    # Initial condition: u(x,0) = exp(5*y)*sin(pi*x)
    u_n.interpolate(lambda X: np.exp(5.0 * X[1]) * np.sin(np.pi * X[0]))
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # 6. Set up variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - kappa * laplacian(u^{n+1}) = f^{n+1}
    # Weak form:
    # integral( u^{n+1} * v / dt ) + integral( kappa * grad(u^{n+1}) . grad(v) )
    #   = integral( u^n * v / dt ) + integral( f^{n+1} * v )
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Expression for interpolating exact BC
    exact_bc_ufl = fem.Expression(u_exact_expr, V.element.interpolation_points)
    
    # Time-stepping loop
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_bc_func.interpolate(exact_bc_ufl)
        bc = fem.dirichletbc(u_bc_func, dofs)
        
        # We need to reassemble A with BCs applied each time since BC values change
        # Actually, for efficiency, reassemble matrix with BCs
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
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_h.x.array[:]
    
    # 7. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on same grid for analysis
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda X: np.exp(5.0 * X[1]) * np.sin(np.pi * X[0]))
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": max(nx, ny),
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