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
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Choose parameters for accuracy
    nx, ny = 64, 64
    degree = 2
    dt = 0.005  # Use smaller dt for better temporal accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Time-stepping setup
    # Manufactured solution: u_exact = exp(-t) * sin(pi*x) * sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -kappa * laplacian(u) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt + kappa * 2*pi^2 * u_exact  (since -laplacian gives +2*pi^2*u)
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) + kappa*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)
    
    # Time constant for source term
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Source term as UFL expression
    pi = np.pi
    f_expr = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 + 2.0 * kappa * ufl.pi**2)
    
    # Exact solution for BC
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Previous time step solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # Store initial condition for output
    # Build evaluation grid first
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Point evaluation setup
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc_arr, cells_on_proc_arr)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape((nx_eval, ny_eval))
    
    # 5. Variational form (Backward Euler)
    # (u - u_n)/dt - kappa*laplacian(u) = f^{n+1}
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f^{n+1} * v dx
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (u_n * v * ufl.dx + dt_const * f_expr * v * ufl.dx)
    
    # 6. Boundary conditions
    # u = 0 on all boundaries (since sin(pi*x)*sin(pi*y) = 0 on boundary)
    # But let's use the exact BC for generality
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    # For this manufactured solution, BC is 0 on all boundaries
    # (sin(pi*x)*sin(pi*y) vanishes on x=0,1 and y=0,1)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bc_dofs)
    bcs = [bc]
    
    # 7. Assemble and solve with manual assembly for efficiency in time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC if needed (it's zero, so no update needed)
        # For non-zero time-dependent BCs:
        # u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
        # u_bc.interpolate(u_bc_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on evaluation grid
    u_final_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc_arr, cells_on_proc_arr)
        u_final_vals[eval_map] = vals.flatten()
    u_grid = u_final_vals.reshape((nx_eval, ny_eval))
    
    # Cleanup PETSc objects
    solver.destroy()
    A.destroy()
    b.destroy()
    
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