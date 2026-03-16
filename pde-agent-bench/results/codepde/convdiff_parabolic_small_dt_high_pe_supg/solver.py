import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec["pde"]
    eps_val = pde["diffusion"]["eps"]
    beta = pde["convection"]["beta"]
    time_cfg = pde["time"]
    t_end = time_cfg["t_end"]
    dt_suggested = time_cfg["dt"]
    
    # Use smaller dt for accuracy with high Peclet number
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution - need fine enough for high Pe
    nx = ny = 100
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 2. Function space - P1 with SUPG
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 3. Define manufactured solution symbolically
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution: u = exp(-t)*sin(4*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = du/dt - eps*laplacian(u) + beta . grad(u)
    # du/dt = -exp(-t)*sin(4*pi*x)*sin(pi*y)
    # grad(u) = exp(-t) * (4*pi*cos(4*pi*x)*sin(pi*y), sin(4*pi*x)*pi*cos(pi*y))
    # laplacian(u) = exp(-t) * (-16*pi^2*sin(4*pi*x)*sin(pi*y) - pi^2*sin(4*pi*x)*sin(pi*y))
    #              = -exp(-t) * 17*pi^2 * sin(4*pi*x)*sin(pi*y)
    
    dudt_ufl = -ufl.exp(-t_const) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f_ufl = dudt_ufl - eps_val * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 4. Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x,0) = sin(4*pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(4 * np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 5. SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_val)
    # Classic SUPG tau
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0, Pe_cell))
    
    # 6. Backward Euler time discretization with SUPG
    # Standard Galerkin part:
    # (u - u_n)/dt - eps*lap(u) + beta.grad(u) = f
    # Weak form: (u - u_n)/dt * v + eps*grad(u).grad(v) + beta.grad(u)*v = f*v
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(eps_val))
    
    # Bilinear form (Galerkin)
    a_gal = (u * v / dt_const) * ufl.dx \
          + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    
    # RHS (Galerkin)
    L_gal = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # SUPG stabilization terms
    # Residual of the strong form applied to trial function:
    # R(u) = u/dt - eps*lap(u) + beta.grad(u) - f - u_n/dt
    # For linear elements, lap(u) = 0 within each element
    # So R(u) ≈ u/dt + beta.grad(u) - f - u_n/dt
    
    # SUPG test function modification: v_supg = tau * beta.grad(v)
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    # SUPG bilinear form
    a_supg = (u / dt_const) * v_supg * ufl.dx \
           + ufl.dot(beta_vec, ufl.grad(u)) * v_supg * ufl.dx
    # Note: for P1, -eps*lap(u) = 0 element-wise, so we skip it
    
    # SUPG RHS
    L_supg = (u_n / dt_const) * v_supg * ufl.dx + f_ufl * v_supg * ufl.dx
    
    a_form = a_gal + a_supg
    L_form = L_gal + L_supg
    
    # 7. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    # BC values will be updated each time step
    
    # 8. Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # 9. Assemble and solve with manual approach for time stepping
    # Solution function
    uh = fem.Function(V)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(comm)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = float(t)
        u_bc.interpolate(lambda X, tv=t_val: np.exp(-tv) * np.sin(4 * np.pi * X[0]) * np.sin(np.pi * X[1]))
        bc = fem.dirichletbc(u_bc, bc_dofs)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_compiled, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        b = petsc.create_vector(L_compiled)
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.setOperators(A)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
        
        # Clean up PETSc objects
        A.destroy()
        b.destroy()
    
    # 10. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also extract initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    solver.destroy()
    
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
            "stabilization": "supg",
        }
    }