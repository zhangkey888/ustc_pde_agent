import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    time_params = pde_config.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Agent-selectable parameters
    N = 80  # mesh resolution
    degree = 2  # element degree
    dt = 0.005  # smaller dt for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # kappa = 1 + 0.3*cos(2*pi*x)*cos(2*pi*y)
    kappa = 1.0 + 0.3 * ufl.cos(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    # Source term f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # grad(u) = exp(-t) * (2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y))
    # We need: f = du/dt - div(kappa * grad(u))
    # Let's compute this symbolically with UFL
    # We define u_exact as a UFL expression and compute f from it
    
    # For the source term, we need to compute div(kappa * grad(u_exact))
    # But u_exact involves SpatialCoordinate, so we can use UFL's diff
    # However, UFL can differentiate spatial coordinates directly via grad
    
    # Actually, u_exact_ufl is already a UFL expression of spatial coords and t_const
    # We can compute grad and div directly
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa * grad_u_exact)
    
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y) = -u_exact
    dudt = -u_exact_ufl
    
    # f = du/dt - div(kappa * grad(u))
    # PDE: du/dt - div(kappa*grad(u)) = f
    # So f = dudt - div_kappa_grad_u
    f_expr = dudt - div_kappa_grad_u
    
    # 4. Define variational forms for backward Euler
    # Backward Euler: (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + (kappa * grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    
    dt_c = fem.Constant(domain, default_scalar_type(dt))
    
    a_form = (u_trial * v_test / dt_c + kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test))) * ufl.dx
    L_form = (u_n * v_test / dt_c + f_expr * v_test) * ufl.dx
    
    # 5. Initial condition: u(x, 0) = sin(2*pi*x)*sin(2*pi*y)
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # 6. Boundary conditions (Dirichlet, time-dependent)
    # g = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    
    # 7. Assemble and solve with manual time-stepping
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (kappa doesn't depend on time, so A is constant)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)  # we'll use petsc vector
    b_vec = petsc.create_vector(L_compiled)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    
    # Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update boundary condition
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on 50x50 uniform grid
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
    
    # Also extract initial condition on grid
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
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