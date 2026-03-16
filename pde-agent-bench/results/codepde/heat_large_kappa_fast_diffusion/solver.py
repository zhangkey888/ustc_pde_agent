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
    kappa = float(coefficients.get("kappa", 5.0))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.08))
    dt_suggested = float(time_params.get("dt", 0.004))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy with large kappa
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution - choose appropriately for accuracy target ~1e-3
    nx, ny = 80, 80
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Manufactured solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(pi*y)
    # laplacian u = -exp(-t)*(4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y) = -5*pi^2*exp(-t)*sin(2*pi*x)*sin(pi*y)
    # -kappa * laplacian u = 5*kappa*pi^2*exp(-t)*sin(2*pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian_u = exp(-t)*sin(2*pi*x)*sin(pi*y)*(-1 + 5*kappa*pi^2)
    
    # Time as a Constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # 4. Define trial/test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Exact solution as UFL expression for BCs and source
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term
    f_coeff = (-1.0 + 5.0 * kappa * pi**2)
    f_expr = f_coeff * ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t^{n+1})
    # Weak form: (u, v)/dt + kappa*(grad u, grad v) = (u_n, v)/dt + (f, v)
    
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # 5. Boundary conditions
    # All boundary: u = g = exact solution at current time
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function that we'll update each time step
    u_bc = fem.Function(V)
    
    # 6. Initial condition: u(x, 0) = sin(2*pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(2 * pi * x[0]) * np.sin(pi * x[1]))
    
    # Store initial condition for output
    # Build evaluation grid first
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Prepare point evaluation infrastructure
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
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    def evaluate_on_grid(func):
        vals = np.full(points.shape[1], np.nan)
        if len(points_on_proc) > 0:
            v = func.eval(points_on_proc_arr, cells_on_proc_arr)
            vals[eval_map] = v.flatten()
        return vals.reshape(nx_eval, ny_eval)
    
    u_initial = evaluate_on_grid(u_n)
    
    # 7. Assemble and solve with manual assembly for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble LHS matrix (constant in time for backward Euler with constant kappa)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
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
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC
        t_val = t
        u_bc.interpolate(lambda x, tv=t_val: np.exp(-tv) * np.sin(2 * pi * x[0]) * np.sin(pi * x[1]))
        
        # Reassemble matrix (BCs might change lifting)
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
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_grid = evaluate_on_grid(u_sol)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
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