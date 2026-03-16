import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    # Extract parameters
    epsilon = pde_config.get("epsilon", 1.0)
    reaction_type = pde_config.get("reaction_type", "linear")
    reaction_coefficient = pde_config.get("reaction_coefficient", 0.0)
    
    # Time parameters
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.005)
    time_scheme = time_params.get("scheme", "crank_nicolson")
    
    # Manufactured solution: u = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y))
    # We need to derive the source term f from this.
    
    # Use a fine mesh and higher-order elements for multi-frequency solution
    nx, ny = 100, 100
    degree = 2
    dt = dt_suggested
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution as UFL expression
    def u_exact_ufl(t_val):
        return ufl.exp(-t_val) * (ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) 
                                   + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1]))
    
    u_exact = u_exact_ufl(t_const)
    
    # Compute source term:
    # du/dt = -exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y)) = -u_exact
    # -eps * laplacian(u) :
    #   laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    #   laplacian(0.2*sin(6*pi*x)*sin(5*pi*y)) = -0.2*(36*pi^2 + 25*pi^2)*sin(6*pi*x)*sin(5*pi*y)
    #                                            = -0.2*61*pi^2*sin(6*pi*x)*sin(5*pi*y)
    #   So -eps*laplacian(u) = eps*exp(-t)*(2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*61*pi^2*sin(6*pi*x)*sin(5*pi*y))
    # R(u) = reaction_coefficient * u (for linear reaction)
    # f = du/dt - eps*laplacian(u) + R(u)
    #   = -u_exact + eps*exp(-t)*(2*pi^2*sin... + 0.2*61*pi^2*sin...) + reaction_coefficient*u_exact
    #   = u_exact*(-1 + reaction_coefficient) + eps*exp(-t)*(2*pi^2*sin(pi*x)*sin(pi*y) + 0.2*61*pi^2*sin(6*pi*x)*sin(5*pi*y))
    
    # More robustly, let's compute it symbolically with UFL
    # du/dt for u = exp(-t)*g(x,y) is -exp(-t)*g(x,y) = -u
    dudt = -u_exact  # time derivative of the exact solution
    
    # Laplacian via UFL: div(grad(u_exact))
    laplacian_u = ufl.div(ufl.grad(u_exact))
    
    # Source term: f = du/dt - eps * laplacian(u) + R(u)
    # For linear reaction: R(u) = reaction_coefficient * u
    if reaction_type == "linear":
        R_u_exact = reaction_coefficient * u_exact
    else:
        R_u_exact = reaction_coefficient * u_exact
    
    f_expr = dudt - epsilon * laplacian_u + R_u_exact
    
    # 4. Define variational forms for Crank-Nicolson
    # CN: (u^{n+1} - u^n)/dt - eps * 0.5*(laplacian(u^{n+1}) + laplacian(u^n)) 
    #     + 0.5*(R(u^{n+1}) + R(u^n)) = 0.5*(f^{n+1} + f^n)
    # Weak form:
    # (u^{n+1}, v)/dt + 0.5*eps*(grad(u^{n+1}), grad(v)) + 0.5*reaction_coeff*(u^{n+1}, v)
    # = (u^n, v)/dt - 0.5*eps*(grad(u^n), grad(v)) - 0.5*reaction_coeff*(u^n, v) + 0.5*(f^{n+1} + f^n, v)
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    eps_c = fem.Constant(domain, default_scalar_type(epsilon))
    react_c = fem.Constant(domain, default_scalar_type(reaction_coefficient))
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # For CN, we need f at n+1 and n
    # t_const will represent t^{n+1}, t_n_const will represent t^n
    t_n_const = fem.Constant(domain, default_scalar_type(0.0))
    
    u_exact_np1 = u_exact_ufl(t_const)  # at t^{n+1}
    u_exact_n = u_exact_ufl(t_n_const)  # at t^n
    
    # Source terms
    dudt_np1 = -u_exact_np1
    laplacian_u_np1 = ufl.div(ufl.grad(u_exact_np1))
    R_np1 = react_c * u_exact_np1
    f_np1 = dudt_np1 - eps_c * laplacian_u_np1 + R_np1
    
    dudt_n = -u_exact_n
    laplacian_u_n = ufl.div(ufl.grad(u_exact_n))
    R_n = react_c * u_exact_n
    f_n = dudt_n - eps_c * laplacian_u_n + R_n
    
    theta = 0.5  # Crank-Nicolson
    
    # Bilinear form (LHS)
    a_form = (ufl.inner(u_trial, v_test) / dt_c * ufl.dx
              + theta * eps_c * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
              + theta * react_c * ufl.inner(u_trial, v_test) * ufl.dx)
    
    # Linear form (RHS)
    L_form = (ufl.inner(u_n, v_test) / dt_c * ufl.dx
              - (1.0 - theta) * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v_test)) * ufl.dx
              - (1.0 - theta) * react_c * ufl.inner(u_n, v_test) * ufl.dx
              + theta * ufl.inner(f_np1, v_test) * ufl.dx
              + (1.0 - theta) * ufl.inner(f_n, v_test) * ufl.dx)
    
    # 5. Boundary conditions - time dependent
    # We need to update BC at each time step
    u_bc_func = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs_bc)
    
    # 6. Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y)
    def u_initial_func(x_arr):
        return (np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]) 
                + 0.2 * np.sin(6 * np.pi * x_arr[0]) * np.sin(5 * np.pi * x_arr[1]))
    
    u_n.interpolate(u_initial_func)
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_compiled)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # 7. Time stepping
    t = 0.0
    n_steps = int(round(t_end / dt))
    dt_actual = t_end / n_steps
    dt_c.value = dt_actual
    
    total_iterations = 0
    
    for step in range(n_steps):
        t_old = t
        t += dt_actual
        
        # Update time constants
        t_const.value = t
        t_n_const.value = t_old
        
        # Update boundary condition
        t_val = t
        u_bc_func.interpolate(lambda x_arr, tv=t_val: (
            np.exp(-tv) * (np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1])
                           + 0.2 * np.sin(6 * np.pi * x_arr[0]) * np.sin(5 * np.pi * x_arr[1]))
        ))
        
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
    
    # 8. Extract solution on 80x80 grid
    nx_out, ny_out = 80, 80
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on grid
    u_init_func_eval = fem.Function(V)
    u_init_func_eval.interpolate(u_initial_func)
    
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func_eval.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
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
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        }
    }