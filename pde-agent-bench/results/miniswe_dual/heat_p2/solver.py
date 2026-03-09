import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None) -> dict:
    """Solve the transient heat equation with manufactured solution."""
    
    # Default parameters from problem description
    kappa = 1.0
    t_end = 0.06
    dt = 0.01
    scheme = "backward_euler"
    
    # Parse case_spec if provided
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        coeffs = pde.get("coefficients", {})
        kappa = coeffs.get("kappa", kappa)
        time_params = pde.get("time", {})
        t_end = time_params.get("t_end", t_end)
        dt = time_params.get("dt", dt)
        scheme = time_params.get("scheme", scheme)
    
    # Choose mesh resolution and element degree for accuracy
    N = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time variable as a Constant
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u_exact = exp(-t) * (x^2 + y^2)
    u_exact_ufl = ufl.exp(-t) * (x[0]**2 + x[1]**2)
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t) * (x^2 + y^2)
    # laplacian(u) = exp(-t) * (2 + 2) = 4*exp(-t)
    # f = -exp(-t)*(x^2+y^2) - kappa * 4 * exp(-t)
    # Wait: the PDE is du/dt - kappa*laplacian(u) = f
    # So f = du/dt - kappa*laplacian(u) = -exp(-t)*(x^2+y^2) - kappa*4*exp(-t)
    f_ufl = ufl.exp(-t) * (-(x[0]**2 + x[1]**2) - 4.0 * kappa)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Current solution
    u_h = fem.Function(V)
    
    # Kappa constant
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Initial condition: u(x, 0) = x^2 + y^2
    u_n.interpolate(fem.Expression(x[0]**2 + x[1]**2, V.element.interpolation_points))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    # => u*v dx + dt*kappa*grad(u).grad(v) dx = u_n*v dx + dt*f*v dx
    a = u * v * ufl.dx + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_c * f_ufl * v * ufl.dx
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # BC function
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for backward Euler with constant kappa and dt)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector using function space
    b_vec = petsc.create_vector(V)
    
    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # Time stepping
    n_steps = int(round(t_end / dt))
    current_time = 0.0
    total_iterations = 0
    
    for step in range(n_steps):
        current_time += dt
        t.value = current_time
        
        # Update boundary condition
        u_bc.interpolate(bc_expr)
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
    
    # Also evaluate initial condition on same grid
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals2.flatten()
    
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b_vec.destroy()
    
    result = {
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
    
    return result


if __name__ == "__main__":
    start = time.time()
    result = solve()
    elapsed = time.time() - start
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    t_end = 0.06
    u_exact = np.exp(-t_end) * (XX**2 + YY**2)
    
    u_computed = result["u"]
    
    # L2-like error on grid
    error = np.sqrt(np.mean((u_computed - u_exact)**2))
    max_error = np.max(np.abs(u_computed - u_exact))
    
    print(f"Solution shape: {u_computed.shape}")
    print(f"Time: {elapsed:.3f}s")
    print(f"NaN count: {np.isnan(u_computed).sum()}")
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
