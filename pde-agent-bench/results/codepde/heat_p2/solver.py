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
    t_end = time_params.get("t_end", 0.06)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Use suggested dt
    dt_val = dt_suggested
    n_steps = int(round(t_end / dt_val))
    dt_val = t_end / n_steps  # adjust to hit t_end exactly
    
    # Mesh and element parameters
    nx = ny = 64
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = exp(-t)*(x^2 + y^2)
    # du/dt = -exp(-t)*(x^2 + y^2)
    # nabla^2 u = exp(-t)*(2 + 2) = 4*exp(-t)
    # f = du/dt - kappa * nabla^2 u = -exp(-t)*(x^2+y^2) - kappa*4*exp(-t)
    #   = exp(-t)*(-x^2 - y^2 - 4*kappa)
    
    # Time constant for updating
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt_val))
    kappa_const = fem.Constant(domain, default_scalar_type(kappa))
    
    # Source term at current time (for backward Euler, evaluate f at t^{n+1})
    # f = exp(-t)*(- x^2 - y^2 - 4*kappa)
    f_expr = ufl.exp(-t_const) * (-(x[0]**2 + x[1]**2) - 4.0 * kappa_const)
    
    # 4. Define variational problem for backward Euler:
    # (u^{n+1} - u^n)/dt - kappa * nabla^2 u^{n+1} = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + kappa*(grad(u^{n+1}), grad(v)) = (f^{n+1}, v) + (u^n, v)/dt
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    
    a = (u * v / dt_const) * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (f_expr * v) * ufl.dx + (u_n * v / dt_const) * ufl.dx
    
    # 5. Initial condition: u(x,0) = x^2 + y^2
    u_n.interpolate(lambda x: x[0]**2 + x[1]**2)
    
    # Store initial condition for output
    # Build evaluation grid first
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_eval * ny_eval))
    points_2d[0, :] = X.ravel()
    points_2d[1, :] = Y.ravel()
    
    # Point evaluation setup
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
    
    points_on_proc = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)
    
    def evaluate_function(func):
        u_values = np.full(nx_eval * ny_eval, np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc, cells_on_proc)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_eval, ny_eval)
    
    # Evaluate initial condition
    u_initial = evaluate_function(u_n)
    
    # 6. Boundary conditions - update at each time step
    # u_exact on boundary = exp(-t)*(x^2 + y^2)
    u_bc = fem.Function(V)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # 7. Time stepping with manual assembly for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem since kappa and dt don't change)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
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
    t_current = 0.0
    
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        
        # Update boundary condition
        t_cur = t_current
        u_bc.interpolate(lambda x, t=t_cur: np.exp(-t) * (x[0]**2 + x[1]**2))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Re-assemble matrix (BCs might change lifting)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
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
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on grid
    u_grid = evaluate_function(u_sol)
    
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
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }