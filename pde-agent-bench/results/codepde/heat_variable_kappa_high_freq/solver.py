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
    dt_suggested = time_params.get("dt", 0.005)
    
    # Use a finer dt for accuracy
    dt = 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact subdivision
    
    # Mesh resolution and element degree
    nx = ny = 80
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, default_scalar_type(0.0))
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    
    pi = np.pi
    
    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # kappa = 1 + 0.3*sin(6*pi*x)*sin(6*pi*y)
    kappa = 1.0 + 0.3 * ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1])
    
    # Source term f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # grad(u) = exp(-t) * (2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y))
    # We compute f symbolically using UFL
    # f = du/dt - div(kappa * grad(u_exact))
    # But we need to be careful: ufl can compute div(kappa * grad(u_exact_ufl))
    
    du_dt = -ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    f_expr = du_dt - ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    # 4. Define variational problem for backward Euler
    # (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    # Weak form: (u^{n+1}, v)/dt + (kappa * grad(u^{n+1}), grad(v)) = (f^{n+1}, v) + (u^n, v)/dt
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step
    
    # Bilinear form (LHS)
    a = (u_trial * v_test / dt_const) * ufl.dx + ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    # Linear form (RHS)
    L = (u_n * v_test / dt_const) * ufl.dx + f_expr * v_test * ufl.dx
    
    # 5. Boundary conditions - Dirichlet from exact solution
    # u_exact on boundary at current time
    u_bc_func = fem.Function(V)
    
    # Create expression for exact solution for interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # 6. Initial condition: u(x, 0) = sin(2*pi*x)*sin(2*pi*y)
    t_const.value = 0.0
    u_n.interpolate(u_exact_expr)
    
    # Compute initial condition on grid for output
    # Build evaluation grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_eval * ny_eval)])
    
    # Save initial condition
    u_initial_vals = _evaluate_function(domain, u_n, points_2d, nx_eval, ny_eval)
    
    # 7. Compile forms and set up manual assembly for time loop
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (kappa doesn't depend on time, so A is constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # 8. Time stepping
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt
        t_const.value = current_t
        
        # Update BC
        u_bc_func.interpolate(u_exact_expr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update u_n for next step
        u_n.x.array[:] = u_h.x.array[:]
    
    # 9. Extract solution on evaluation grid
    u_grid = _evaluate_function(domain, u_h, points_2d, nx_eval, ny_eval)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_vals,
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


def _evaluate_function(domain, u_func, points_3d, nx_eval, ny_eval):
    """Evaluate a dolfinx Function on a set of points and return as (nx, ny) array."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    n_points = points_3d.shape[1]
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape(nx_eval, ny_eval)