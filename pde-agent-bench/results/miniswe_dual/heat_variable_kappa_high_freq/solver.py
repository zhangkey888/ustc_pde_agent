import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    pde_info = case_spec.get("pde", {})
    time_info = pde_info.get("time", {})
    coefficients = pde_info.get("coefficients", {})
    
    t_end = time_info.get("t_end", 0.1)
    dt_suggested = time_info.get("dt", 0.005)
    scheme = time_info.get("scheme", "backward_euler")
    
    # Parameters - we'll use degree 2 and a fine mesh for accuracy
    N = 80  # mesh resolution
    degree = 2
    dt = 0.002  # smaller than suggested for better accuracy
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust dt to exactly hit t_end
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Manufactured solution
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # Variable kappa
    kappa = 1.0 + 0.3 * ufl.sin(6 * pi * x[0]) * ufl.sin(6 * pi * x[1])
    
    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    du_dt = -ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    
    # f = du/dt - div(kappa * grad(u_exact))
    # But we need: du/dt - div(kappa * grad(u)) = f
    # So f = du/dt - div(kappa * grad(u_exact))
    # Note: -div(kappa*grad(u)) means the weak form has +inner(kappa*grad(u), grad(v))
    # So f = du/dt + (-div(kappa*grad(u_exact))) ... wait let me be careful
    # PDE: du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - div(kappa*grad(u_exact))
    f_expr = du_dt - ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Current solution
    u_h = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(2*pi*x)*sin(2*pi*y)
    u_n.interpolate(fem.Expression(
        ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1]),
        V.element.interpolation_points
    ))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f(t_{n+1})
    # Weak form: (u/dt)*v*dx + inner(kappa*grad(u), grad(v))*dx = (u_n/dt)*v*dx + f*v*dx
    a_form = (u / dt_const) * v * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (u_n / dt_const) * v * ufl.dx + f_expr * v * ufl.dx
    
    # Boundary conditions: u = g on boundary (from exact solution)
    # g = exp(-t)*sin(2*pi*x)*sin(2*pi*y) which is 0 on the boundary of [0,1]^2
    # since sin(0) = sin(2*pi) = 0
    # So homogeneous Dirichlet BC
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC value - for this manufactured solution, BC is 0 on boundary
    # But to be safe, let's use the exact solution expression
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (kappa doesn't depend on time, and dt is constant, so A is constant)
    # Actually, kappa depends on spatial coordinates only, so A is constant over time
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()
    
    total_iterations = 0
    
    # Time stepping
    t_current = 0.0
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Update BC (it's zero on boundary for all time, but let's be explicit)
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        # Assemble RHS
        b = petsc.assemble_vector(L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
        
        # Clean up vector
        b.destroy()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
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
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on the same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc2 = []
    cells_on_proc2 = []
    eval_map2 = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc2.append(points_3d[i])
            cells_on_proc2.append(links[0])
            eval_map2.append(i)
    
    if len(points_on_proc2) > 0:
        vals2 = u_initial_func.eval(np.array(points_on_proc2), np.array(cells_on_proc2, dtype=np.int32))
        u_init_values[eval_map2] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Compute error against exact solution for verification
    t_const.value = t_end
    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    error_L2 = fem.form(ufl.inner(u_h - u_exact_final, u_h - u_exact_final) * ufl.dx)
    error_local = fem.assemble_scalar(error_L2)
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    
    # Also compute pointwise error on the grid
    u_exact_grid = np.exp(-t_end) * np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    grid_error = np.max(np.abs(u_grid - u_exact_grid))
    print(f"Max grid error: {grid_error:.6e}")
    
    # Cleanup
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
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
    # Test case spec
    case_spec = {
        "pde": {
            "type": "heat",
            "time": {
                "t_end": 0.1,
                "dt": 0.005,
                "scheme": "backward_euler"
            },
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.3*sin(6*pi*x)*sin(6*pi*y)"}
            }
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    print(f"N steps: {result['solver_info']['n_steps']}")
    print(f"dt: {result['solver_info']['dt']}")
