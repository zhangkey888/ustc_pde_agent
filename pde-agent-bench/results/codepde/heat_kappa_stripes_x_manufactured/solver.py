import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a smaller dt for accuracy
    dt = 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # Mesh resolution
    nx = ny = 80
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # 4. Manufactured solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Diffusivity: kappa = 1 + 0.5*sin(6*pi*x)
    kappa = 1.0 + 0.5 * ufl.sin(6 * pi * x[0])
    
    # Compute source term f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(pi*y) = -u_exact
    # We need to compute div(kappa * grad(u_exact)) symbolically
    # grad(u_exact) = exp(-t) * [2*pi*cos(2*pi*x)*sin(pi*y), pi*sin(2*pi*x)*cos(pi*y)]
    # div(kappa * grad(u_exact)) needs chain rule with kappa(x)
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    dudt = -u_exact_ufl  # d/dt [exp(-t)*...] = -exp(-t)*...
    
    # f = du/dt - div(kappa * grad(u))
    f_expr = dudt - ufl.div(kappa * grad_u_exact)
    
    # 5. Define variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + (kappa * grad(u^{n+1}), grad(v)) = (f^{n+1}, v) + (u^n, v)/dt
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Bilinear form (LHS)
    a = (u_trial * v_test / dt_const) * ufl.dx + ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    # Linear form (RHS)
    L = (u_n * v_test / dt_const) * ufl.dx + f_expr * v_test * ufl.dx
    
    # 6. Boundary conditions
    # u = g = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function that we'll update each time step
    u_bc = fem.Function(V)
    
    # Create expression for exact solution for interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Initial condition: u(x,0) = sin(2*pi*x)*sin(pi*y) (exp(0)=1)
    t_const.value = 0.0
    u_n.interpolate(u_exact_expr)
    
    # Store initial condition for output
    # We'll evaluate it on the grid later
    
    # 8. Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.petsc.create_vector(L_form)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    # 9. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC
        u_bc.interpolate(u_exact_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for Dirichlet BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 10. Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_eval * ny_eval))
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
    
    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Also evaluate initial condition
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)
    
    u_init_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_eval, ny_eval))
    
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
        }
    }