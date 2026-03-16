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
    t_end = time_params.get("t_end", 0.06)
    dt_suggested = time_params.get("dt", 0.01)
    
    # Use smaller dt for accuracy with P2 elements
    dt = 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division
    
    # Mesh resolution
    nx = ny = 64
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2 for better accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # 4. Time-dependent manufactured solution: u_exact = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # kappa = 1 + 0.4*sin(2*pi*x)*sin(2*pi*y)
    
    kappa_ufl = 1.0 + 0.4 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Time parameter as a Constant so we can update it
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    
    # Exact solution as UFL expression (for BC and source)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Compute source term f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # We need to compute -div(kappa * grad(u_exact)) symbolically
    # Let's define u_exact as a UFL expression and compute the source
    
    # grad(u_exact) 
    # u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # u_x = exp(-t)*2*pi*cos(2*pi*x)*sin(2*pi*y)
    # u_y = exp(-t)*sin(2*pi*x)*2*pi*cos(2*pi*y)
    
    # For the source term, we need:
    # f = du/dt - div(kappa * grad(u))
    # du/dt = -u_exact
    
    # We'll compute div(kappa * grad(u_exact)) using UFL's symbolic differentiation
    # But u_exact_ufl depends on t_const which is a Constant, not a spatial variable
    # UFL can compute grad w.r.t. spatial coords
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    flux = kappa_ufl * grad_u_exact
    div_flux = ufl.div(flux)
    
    # f = du/dt - div(kappa * grad(u)) => f = -u_exact - div(kappa * grad(u_exact))
    # Wait: the PDE is du/dt - div(kappa * grad(u)) = f
    # So f = du/dt - div(kappa * grad(u_exact))
    # du/dt of exact = -exp(-t)*sin(2*pi*x)*sin(2*pi*y) = -u_exact
    
    f_ufl = -u_exact_ufl - div_flux
    
    # 5. Set up variational forms for backward Euler
    # (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + (kappa * grad(u^{n+1}), grad(v)) = (f^{n+1}, v) + (u^n, v)/dt
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    u_n = fem.Function(V)  # solution at previous time step
    
    # Bilinear form
    a = (u_trial * v_test / dt_const) * ufl.dx + ufl.inner(kappa_ufl * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    # Linear form
    L = (u_n * v_test / dt_const) * ufl.dx + f_ufl * v_test * ufl.dx
    
    # 6. Boundary conditions - Dirichlet from exact solution
    # We need to update BC at each time step
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(2*pi*x)*sin(2*pi*y)
    u_n.interpolate(lambda x: np.sin(2 * pi * x[0]) * np.sin(2 * pi * x[1]))
    
    # Store initial condition for output
    # Create evaluation grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((3, points_2d.shape[1]))
    points_3d[:2, :] = points_2d
    
    # Evaluate initial condition
    def evaluate_function(u_func, points_3d_arr):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d_arr.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d_arr.T)
        
        n_points = points_3d_arr.shape[1]
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(n_points):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points_3d_arr[:, i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full(n_points, np.nan)
        if len(points_on_proc) > 0:
            pts = np.array(points_on_proc)
            cls = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts, cls)
            u_values[eval_map] = vals.flatten()
        return u_values
    
    u_initial_vals = evaluate_function(u_n, points_3d)
    u_initial_grid = u_initial_vals.reshape((nx_eval, ny_eval))
    
    # 7. Time stepping with manual assembly for efficiency
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Since kappa doesn't depend on time, A is constant
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    u_sol = fem.Function(V)
    
    # Set up KSP solver
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update BC
        t_val = float(t_current)
        u_bc_func.interpolate(
            lambda x, tv=t_val: np.exp(-tv) * np.sin(2 * pi * x[0]) * np.sin(2 * pi * x[1])
        )
        
        # Reassemble A (BCs may change lifting)
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
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # 8. Extract solution on evaluation grid
    u_values = evaluate_function(u_sol, points_3d)
    u_grid = u_values.reshape((nx_eval, ny_eval))
    
    # Clean up PETSc objects
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