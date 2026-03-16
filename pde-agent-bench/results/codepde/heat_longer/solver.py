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
    kappa = coefficients.get("kappa", 0.5)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = 0.005
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to exactly hit t_end
    
    # Mesh resolution and element degree
    nx = ny = 80
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    # Manufactured solution: u = exp(-2*t)*cos(pi*x)*cos(pi*y)
    pi = np.pi
    
    # Exact solution as UFL expression (for BCs and source)
    u_exact_ufl = ufl.exp(-2 * t_const) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -2 * exp(-2t) * cos(pi*x) * cos(pi*y)
    # laplacian(u) = -2*pi^2 * exp(-2t) * cos(pi*x) * cos(pi*y)
    # f = -2*exp(-2t)*cos(pi*x)*cos(pi*y) - kappa*(-2*pi^2)*exp(-2t)*cos(pi*x)*cos(pi*y)
    # f = exp(-2t)*cos(pi*x)*cos(pi*y) * (-2 + 2*kappa*pi^2)
    f_expr = ufl.exp(-2 * t_const) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) * (-2.0 + 2.0 * kappa_const * ufl.pi**2)
    
    # 4. Initial condition: u(x, 0) = cos(pi*x)*cos(pi*y)
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_n.interpolate(lambda X: np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))
    
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
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    # Evaluate initial condition
    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_initial_vals[eval_map] = vals.flatten()
    u_initial_grid = u_initial_vals.reshape((nx_eval, ny_eval))
    
    # 5. Variational problem (Backward Euler)
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = u * v * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_expr * v * ufl.dx
    
    # 6. Boundary conditions (time-dependent)
    # We need to update BC at each time step
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    # Set initial BC values at t=0 (will be updated in time loop)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 7. Assemble and solve with manual approach for time stepping
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t = 0.0
    
    # Create expression for BC interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_bc.interpolate(u_exact_expr)
        
        # Reassemble A (since f depends on t through dt_const and t_const, but a doesn't change)
        # Actually a doesn't depend on t, so we don't need to reassemble A
        # But L depends on t through f_expr and u_n
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on evaluation grid
    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(points_on_proc, cells_on_proc)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
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
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }