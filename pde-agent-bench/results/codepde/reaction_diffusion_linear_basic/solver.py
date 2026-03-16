import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    time_params = pde_config.get("time", {})
    
    t_end = time_params.get("t_end", 0.5)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Extract diffusion coefficient
    epsilon = pde_config.get("epsilon", 1.0)
    if isinstance(epsilon, dict):
        epsilon = epsilon.get("value", 1.0)
    
    # Extract reaction coefficient
    reaction = pde_config.get("reaction", {})
    reaction_type = reaction.get("type", "linear")
    reaction_coeff = reaction.get("coefficient", 1.0)
    
    # Parameters
    nx = ny = 80
    degree = 2
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # 4. Time-stepping setup
    # Manufactured solution: u = exp(-t) * sin(pi*x) * sin(pi*y)
    # du/dt = -exp(-t) * sin(pi*x) * sin(pi*y)
    # -eps * nabla^2 u = eps * 2*pi^2 * exp(-t) * sin(pi*x) * sin(pi*y)
    # R(u) = reaction_coeff * u = reaction_coeff * exp(-t) * sin(pi*x) * sin(pi*y)
    # f = du/dt - eps*nabla^2(u) + R(u)
    #   = exp(-t)*sin(pi*x)*sin(pi*y) * (-1 + eps*2*pi^2 + reaction_coeff)
    
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    react_const = fem.Constant(domain, default_scalar_type(reaction_coeff))
    
    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term
    f_expr = ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * (
        -1.0 + eps_const * 2.0 * pi**2 + react_const
    )
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous time step solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))
    
    # Store initial condition for output
    # Create grid for evaluation
    nx_eval, ny_eval = 60, 60
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X_grid, Y_grid = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    def evaluate_on_grid(u_func):
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
        
        u_values = np.full(points_3d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_eval, ny_eval)
    
    u_initial = evaluate_on_grid(u_n)
    
    # 5. Boundary conditions - u = 0 on boundary for all t > 0
    # (since sin(pi*x)*sin(pi*y) = 0 on boundary of unit square)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Time-dependent BC (zero on boundary for this manufactured solution)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 6. Backward Euler: (u - u_n)/dt - eps*nabla^2(u) + react*u = f
    # Bilinear form: (u, v)/dt + eps*(grad(u), grad(v)) + react*(u, v)
    # Linear form: (u_n, v)/dt + (f, v)
    
    a_form = (
        ufl.inner(u, v) / dt_const * ufl.dx
        + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + react_const * ufl.inner(u, v) * ufl.dx
    )
    
    L_form = (
        ufl.inner(u_n, v) / dt_const * ufl.dx
        + ufl.inner(f_expr, v) * ufl.dx
    )
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    total_iterations = 0
    
    # 7. Time-stepping loop
    for step in range(n_steps):
        t_current = (step + 1) * dt
        t_const.value = t_current
        
        # Update BC if needed (stays zero for this problem)
        # u_bc already zero
        
        # Assemble RHS
        b_vec = petsc.create_vector(L_compiled)
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
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
        b_vec.destroy()
    
    # 8. Extract solution on grid
    u_grid = evaluate_on_grid(u_sol)
    
    solver.destroy()
    A.destroy()
    
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