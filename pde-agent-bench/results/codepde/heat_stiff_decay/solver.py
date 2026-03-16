import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    time_params = pde.get("time", {})
    
    kappa = float(coeffs.get("kappa", 0.5))
    t_end = float(time_params.get("t_end", 0.12))
    dt_suggested = float(time_params.get("dt", 0.006))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use a finer dt for accuracy
    dt = dt_suggested
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh resolution
    nx = ny = 64
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    degree = 1
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Manufactured solution: u = exp(-10*t)*sin(pi*x)*sin(pi*y)
    pi = np.pi
    u_exact_ufl = ufl.exp(-10.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # u = exp(-10t)*sin(pi*x)*sin(pi*y)
    # du/dt = -10*exp(-10t)*sin(pi*x)*sin(pi*y)
    # laplacian(u) = -2*pi^2*exp(-10t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) = exp(-10t)*sin(pi*x)*sin(pi*y)*(-10 + kappa*2*pi^2)
    f_ufl = ufl.exp(-10.0 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-10.0 + kappa * 2.0 * ufl.pi**2)
    
    # 4. Define variational problem for backward Euler
    # (u^{n+1} - u^n)/dt - kappa * laplacian(u^{n+1}) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (u^n, v)/dt + (f^{n+1}, v)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx
    
    # 5. Boundary conditions - u = g on boundary (exact solution)
    # For each time step, we need to update the BC
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Set initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Store initial condition for output
    # Evaluate on grid
    nx_eval = ny_eval = 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
    # Build point evaluation infrastructure
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)
    
    def evaluate_on_grid(func):
        vals = np.full(points.shape[1], np.nan)
        if len(points_on_proc) > 0:
            v = func.eval(points_on_proc_arr, cells_on_proc_arr)
            for idx, global_idx in enumerate(eval_map):
                vals[global_idx] = v[idx, 0]
        return vals.reshape((nx_eval, ny_eval))
    
    u_initial = evaluate_on_grid(u_n)
    
    # 7. Compile forms and set up manual assembly for time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.rtol = 1e-10
    solver.atol = 1e-12
    solver.max_it = 1000
    solver.setUp()
    
    uh = fem.Function(V)
    
    total_iterations = 0
    
    # 8. Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        t_val = float(t)
        u_bc.interpolate(lambda x, tv=t_val: np.exp(-10.0 * tv) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting and BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = uh.x.array[:]
    
    # 9. Evaluate final solution on grid
    u_grid = evaluate_on_grid(uh)
    
    # Clean up PETSc objects
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
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }