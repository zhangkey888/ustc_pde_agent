import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

def solve(case_spec: dict) -> dict:
    """Solve reaction-diffusion equation with manufactured solution."""
    
    # Parse case_spec
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    domain_spec = case_spec.get("domain", {})
    
    # Time parameters - hardcoded defaults as specified
    t_end = float(time_spec.get("t_end", 0.4))
    dt_val = float(time_spec.get("dt", 0.01))
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True  # forced since problem description mentions t_end
    
    # Output grid
    output_spec = case_spec.get("output", {})
    nx_out = int(output_spec.get("nx", 60))
    ny_out = int(output_spec.get("ny", 60))
    
    # Solver parameters - adaptive
    mesh_resolution = 64
    element_degree = 2
    ksp_type_str = "gmres"
    pc_type_str = "hypre"
    rtol_val = 1e-10
    
    comm = MPI.COMM_WORLD
    
    # Create quadrilateral mesh on unit square
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, 
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space - use "Lagrange" on quads (which maps to Q elements)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    
    # Time as a constant that we update
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * exp(x) * sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi * x[1])
    
    # Diffusion coefficient epsilon = 1
    epsilon = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Reaction term R(u) = u (linear reaction)
    # We'll compute f from the manufactured solution
    # PDE: du/dt - eps * laplacian(u) + R(u) = f
    # => f = du/dt - eps * laplacian(u) + u
    # du/dt = -exp(-t)*exp(x)*sin(pi*y) = -u_exact
    # laplacian(u) = exp(-t)*(exp(x)*sin(pi*y) - pi^2*exp(x)*sin(pi*y))
    #              = exp(-t)*exp(x)*sin(pi*y)*(1 - pi^2)
    #              = u_exact * (1 - pi^2)
    # So: f = -u_exact - eps * u_exact * (1 - pi^2) + u_exact
    #       = -u_exact - u_exact*(1 - pi^2) + u_exact
    #       = u_exact * (-1 - 1 + pi^2 + 1)
    #       = u_exact * (pi^2 - 1)
    # Let's compute it symbolically with UFL to be safe
    
    # du_dt for manufactured solution
    du_dt_exact = -ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi * x[1])
    
    # Laplacian of u_exact
    # grad(u_exact) = (exp(-t)*exp(x)*sin(pi*y), exp(-t)*exp(x)*pi*cos(pi*y))
    # laplacian = exp(-t)*exp(x)*sin(pi*y) + exp(-t)*exp(x)*(-pi^2)*sin(pi*y)
    #           = exp(-t)*exp(x)*sin(pi*y)*(1 - pi^2)
    # But let's use UFL's div(grad(...)) - we can't directly since u_exact_ufl is not a TrialFunction
    # Instead, compute manually:
    laplacian_u_exact = ufl.exp(-t_const) * ufl.exp(x[0]) * ufl.sin(pi * x[1]) * (1.0 - pi**2)
    
    # R(u) = u (linear reaction)
    R_u_exact = u_exact_ufl
    
    # Source term: f = du/dt - eps * laplacian(u) + R(u)
    f_expr = du_dt_exact - epsilon * laplacian_u_exact + R_u_exact
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Solution function
    u_h = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")  # previous time step
    
    # dt constant
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    
    # Initial condition: u(x, 0) = exp(0)*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)
    t_const.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)
    
    # Store initial condition
    # Create output grid points
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing='ij')
    points_2d = np.column_stack([X_out.ravel(), Y_out.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + R(u) = f
    # => (u - u_n)/dt + eps*(-laplacian(u)) + u = f
    # Weak form: (u/dt)*v + eps*grad(u).grad(v) + u*v = (u_n/dt)*v + f*v
    # Bilinear form (LHS)
    a_form = (u / dt_c) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
    
    # Linear form (RHS)
    L_form = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx
    
    # Boundary conditions - Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    bc_func = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, dofs)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble matrix (constant since linear problem with constant coefficients)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    # Create RHS vector
    b = fem.Function(V)
    b_vec = b.x.petsc_vec
    
    # Setup KSP solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type_str)
    pc = solver.getPC()
    pc.setType(pc_type_str)
    solver.setTolerances(rtol=rtol_val, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(round(t_end / dt_val))
    t_current = 0.0
    total_iterations = 0
    nonlinear_iterations = []
    
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        
        # Update boundary condition
        bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        its = solver.getIterationNumber()
        total_iterations += its
        nonlinear_iterations.append(1)  # linear problem, 1 "Newton" iteration
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate solution on output grid
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    # Also evaluate initial condition for output
    # Reset time to 0 for initial condition evaluation
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    
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
        vals2 = u_init_func.eval(np.array(points_on_proc2), np.array(cells_on_proc2, dtype=np.int32))
        u_init_values[eval_map2] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    # Test with default case_spec
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "time": {
                "t_end": 0.4,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        },
        "domain": {
            "type": "unit_square",
            "x_range": [0, 1],
            "y_range": [0, 1]
        },
        "output": {
            "nx": 60,
            "ny": 60
        }
    }
    
    start = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - start
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Total iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution at t_end
    t_end = 0.4
    nx_out, ny_out = 60, 60
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end) * np.exp(X_out) * np.sin(np.pi * Y_out)
    
    # L2 error on grid
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    linf_error = np.max(np.abs(result['u'] - u_exact))
    print(f"L2 grid error: {error:.6e}")
    print(f"Linf grid error: {linf_error:.6e}")
    print(f"Target error: 8.45e-03")
    print(f"PASS: {error < 8.45e-3}")
