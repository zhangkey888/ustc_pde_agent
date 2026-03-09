import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None) -> dict:
    # Parse case_spec or use defaults
    if case_spec is not None:
        kappa_val = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 10.0))
        time_params = case_spec.get("pde", {}).get("time", {})
        t_end = float(time_params.get("t_end", 0.05))
        dt_suggested = float(time_params.get("dt", 0.005))
        scheme = time_params.get("scheme", "backward_euler")
        nx_eval = case_spec.get("output", {}).get("nx", 50)
        ny_eval = case_spec.get("output", {}).get("ny", 50)
    else:
        kappa_val = 10.0
        t_end = 0.05
        dt_suggested = 0.005
        scheme = "backward_euler"
        nx_eval = 50
        ny_eval = 50

    # Parameters tuned for accuracy target 4.64e-04
    N = 80
    degree = 2
    dt = 0.001  # small dt for temporal accuracy

    comm = MPI.COMM_WORLD

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time and spatial coordinates
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    # Exact solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = du/dt + kappa * (-laplacian(u))  ... wait, let me re-derive
    # PDE: du/dt - div(kappa * grad(u)) = f
    # u = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # grad(u) = exp(-t)*(pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # div(kappa*grad(u)) = kappa * exp(-t) * (-pi^2*sin(pi*x)*sin(pi*y) - pi^2*sin(pi*x)*sin(pi*y))
    #                     = -2*kappa*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - div(kappa*grad(u))
    #   = -exp(-t)*sin(pi*x)*sin(pi*y) - (-2*kappa*pi^2*exp(-t)*sin(pi*x)*sin(pi*y))
    #   = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * (-1.0 + 2.0 * kappa_val * pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Interpolate initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x_arr: np.sin(pi * x_arr[0]) * np.sin(pi * x_arr[1]))
    
    # Solution function
    u_sol = fem.Function(V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u/dt)*v dx + kappa * grad(u).grad(v) dx = (u_n/dt)*v dx + f*v dx
    a = (u * v / dt_const) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_const) * v * ufl.dx + f_ufl * v * ufl.dx
    
    # Boundary conditions
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Create vector using function space
    b = petsc.create_vector(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12)
    
    # Time stepping
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    current_t = 0.0
    
    for step in range(n_steps):
        current_t += dt
        t.value = current_t
        
        # Update BC
        u_bc.interpolate(u_exact_expr)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        solver.setOperators(A)
        
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
        
        A.destroy()
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x_arr: np.sin(pi * x_arr[0]) * np.sin(pi * x_arr[1]))
    
    # Evaluate on grid
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X, Y = np.meshgrid(x_eval, y_eval, indexing='ij')
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    
    u_grid = u_values.reshape((nx_eval, ny_eval))
    u_init_grid = u_init_values.reshape((nx_eval, ny_eval))
    
    # Compute error for verification
    u_exact_vals = np.exp(-t_end) * np.sin(pi * X) * np.sin(pi * Y)
    error = np.sqrt(np.mean((u_grid - u_exact_vals)**2))
    print(f"Grid L2 error: {error:.6e}")
    print(f"N={N}, degree={degree}, dt={dt}, n_steps={n_steps}")
    print(f"Total KSP iterations: {total_iterations}")
    
    solver.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
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


if __name__ == "__main__":
    start = time.time()
    result = solve()
    elapsed = time.time() - start
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6e}, {np.nanmax(result['u']):.6e}]")
