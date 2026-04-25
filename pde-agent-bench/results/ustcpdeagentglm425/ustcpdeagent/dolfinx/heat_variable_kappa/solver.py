import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    time_params = case_spec["pde"]["time"]
    t0 = time_params["t0"]
    t_end = time_params["t_end"]
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Solver parameters
    mesh_res = 128
    element_degree = 2
    dt = 0.002
    time_scheme = "backward_euler"
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Variable kappa
    kappa_expr = 1.0 + 0.3 * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])
    
    # Time variable
    t_var = fem.Constant(domain, ScalarType(t0))
    
    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_exact = ufl.exp(-t_var) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Time derivative: du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y) = -u_exact
    du_dt = -u_exact
    
    # Source term: f = du/dt - div(kappa * grad(u))
    f_expr = du_dt - ufl.div(kappa_expr * ufl.grad(u_exact))
    
    # Functions for source term and BC
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    g_bc = fem.Function(V)
    g_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    # Boundary condition
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_bc, boundary_dofs)
    
    # Bilinear form: (u, v) + dt * (kappa * grad(u), grad(v))
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(u_trial, v_test) * ufl.dx + dt * ufl.inner(kappa_expr * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_compiled = fem.form(a)
    
    # Assemble matrix (time-independent since kappa doesn't depend on time)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
    
    # Initial condition: u(x,0) = sin(2*pi*x)*sin(2*pi*y)
    u_n = fem.Function(V)
    u_exact_init = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u_n.interpolate(fem.Expression(u_exact_init, V.element.interpolation_points))
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Time stepping
    t = t0
    n_steps = 0
    total_iterations = 0
    
    while t < t_end - 1e-12:
        t_new = min(t + dt, t_end)
        t_var.value = ScalarType(t_new)
        
        # Update boundary condition
        g_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        # Update source term
        f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        
        # Build RHS: (u_n + dt*f, v)
        L = ufl.inner(u_n, v_test) * ufl.dx + dt * ufl.inner(f, v_test) * ufl.dx
        L_compiled = fem.form(L)
        
        b = petsc.create_vector(L_compiled.function_spaces)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        
        # Apply lifting for BCs
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update
        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new
        n_steps += 1
    
    # Compute L2 error for verification
    t_var.value = ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_local = fem.assemble_scalar(fem.form(error_expr))
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_global:.6e}")
        print(f"n_steps: {n_steps}, total_iterations: {total_iterations}")
    
    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_global_vals = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_global_vals, op=MPI.MAX)
    u_grid = u_global_vals.reshape(ny_out, nx_out)
    
    # Sample initial condition
    u_init_vals = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        u_init_interp = fem.Function(V)
        u_init_interp.interpolate(
            fem.Expression(ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1]), V.element.interpolation_points)
        )
        vals_init = u_init_interp.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals_init.flatten()
    
    u_init_global = np.zeros_like(u_init_vals)
    comm.Allreduce(u_init_vals, u_init_global, op=MPI.MAX)
    u_initial = u_init_global.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }
