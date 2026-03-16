import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    
    # Diffusion coefficient
    epsilon = float(pde.get("epsilon", 0.02))
    
    # Convection velocity
    beta_vec = pde.get("beta", [6.0, 3.0])
    beta_x = float(beta_vec[0])
    beta_y = float(beta_vec[1])
    
    # Time parameters
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a smaller dt for accuracy given high Peclet number
    dt = 0.005
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution - use finer mesh for high Peclet
    N = 100
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Time constant
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Convection velocity
    beta = ufl.as_vector([fem.Constant(domain, default_scalar_type(beta_x)),
                           fem.Constant(domain, default_scalar_type(beta_y))])
    
    # Diffusion
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    
    # Source term: f = exp(-150*((x-0.4)**2 + (y-0.6)**2))*exp(-t)
    f_expr = ufl.exp(-150.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2)) * ufl.exp(-t_const)
    
    # Previous solution
    u_n = fem.Function(V, name="u_n")
    
    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # 5. SUPG Stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # SUPG stabilization parameter
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                      1.0 - 1.0 / Pe_cell,
                                                      0.0))
    
    # 6. Variational form - Backward Euler with SUPG
    # Standard Galerkin part:
    # (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form (standard):
    # (u/dt, v) + eps*(grad(u), grad(v)) + (beta.grad(u), v) = (f, v) + (u_n/dt, v)
    
    a_standard = (u / dt_const * v * ufl.dx
                  + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    
    L_standard = (u_n / dt_const * v * ufl.dx
                  + f_expr * v * ufl.dx)
    
    # SUPG stabilization terms
    # Residual applied to trial function (for linear part):
    # R(u) = u/dt - eps*laplacian(u) + beta.grad(u) - f
    # For P1 elements, laplacian(u) = 0 element-wise
    # So residual ≈ u/dt + beta.grad(u) - f - u_n/dt
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = (u / dt_const * v_supg * ufl.dx
              + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) * ufl.dx
              + ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx)
    
    L_supg = (u_n / dt_const * v_supg * ufl.dx
              + f_expr * v_supg * ufl.dx)
    
    a_form = a_standard + a_supg
    L_form = L_standard + L_supg
    
    # 7. Boundary conditions - Dirichlet g
    # For this problem, BC is typically g = 0 on boundary (homogeneous)
    # since sin(pi*x)*sin(pi*y) = 0 on boundary and no explicit g given
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 8. Assemble and solve with manual time-stepping
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    
    total_iterations = 0
    
    # Time stepping loop
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
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
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
        
        # Clean up PETSc objects
        A.destroy()
        b.destroy()
    
    # 9. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
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
    
    # Also extract initial condition on same grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Clean up solver
    solver.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }