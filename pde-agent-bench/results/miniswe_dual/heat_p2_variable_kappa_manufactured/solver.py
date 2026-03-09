import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with variable kappa."""
    
    start_time = time.time()
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    time_params = pde.get("time", {})
    domain_spec = case_spec.get("domain", {})
    
    # Time parameters with hardcoded defaults
    t_end = time_params.get("t_end", 0.06)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")
    is_transient = True
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    mesh_resolution = 48
    element_degree = 2
    dt = dt_suggested
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant (will be updated)
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Variable kappa: 1 + 0.4*sin(2*pi*x)*sin(2*pi*y)
    kappa = 1.0 + 0.4 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # We need f such that du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - div(kappa*grad(u))
    dudt_exact = -ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa * grad_u_exact)
    f = dudt_exact - div_kappa_grad_u
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution function
    u_sol = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    
    # Initial condition
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    # Backward Euler weak form:
    # (u - u_n)/dt - div(kappa*grad(u)) = f
    # Weak form: (u/dt)*v*dx + inner(kappa*grad(u), grad(v))*dx = (u_n/dt)*v*dx + f*v*dx
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = (u * v / dt_const) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = (u_n * v / dt_const) * ufl.dx + f * v * ufl.dx
    
    a_compiled = fem.form(a)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (will be reassembled each step since BCs change)
    A = petsc.assemble_matrix(a_compiled, bcs=[fem.dirichletbc(u_bc, boundary_dofs)])
    A.assemble()
    
    # Create RHS vector - use the function space V
    b = petsc.create_vector(V)
    
    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update boundary condition
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Reassemble matrix
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on output grid
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also compute initial condition on grid
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    elapsed = time.time() - start_time
    print(f"Solve completed in {elapsed:.3f}s, {n_steps} steps, {total_iterations} total KSP iterations")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
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
    case_spec = {
        "pde": {
            "type": "heat",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.4*sin(2*pi*x)*sin(2*pi*y)"}
            },
            "time": {
                "t_end": 0.06,
                "dt": 0.01,
                "scheme": "backward_euler"
            },
            "manufactured_solution": "exp(-t)*sin(2*pi*x)*sin(2*pi*y)"
        },
        "domain": {"type": "unit_square", "dim": 2},
        "output": {"nx": 50, "ny": 50, "field": "scalar"}
    }
    
    result = solve(case_spec)
    u_grid = result["u"]
    
    t_end = 0.06
    x_grid = np.linspace(0, 1, 50)
    y_grid = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"L2 grid error: {error:.6e}")
    print(f"Max grid error: {max_error:.6e}")
    print(f"Target: error <= 1.51e-03")
    print(f"Pass: {error <= 1.51e-3}")
