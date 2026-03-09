import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict = None):
    """Solve the transient heat equation."""
    
    # Default parameters from the problem description
    kappa = 0.5
    t_end = 0.2
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
    
    N = 64
    element_degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter as a Constant
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = exp(-2*t)*sin(pi*x)*sin(pi*y)
    pi_val = np.pi
    
    # Source term f = du/dt - kappa * laplacian(u)
    # du/dt = -2 * exp(-2t) * sin(pi*x) * sin(pi*y)
    # laplacian(u) = -2*pi^2 * exp(-2t) * sin(pi*x) * sin(pi*y)
    # f = du/dt + kappa * 2*pi^2 * exp(-2t) * sin(pi*x) * sin(pi*y)
    # f = exp(-2t)*sin(pi*x)*sin(pi*y) * (-2 + 2*kappa*pi^2)
    f_coeff = 2.0 * kappa * pi_val**2 - 2.0
    f_expr_ufl = f_coeff * ufl.exp(-2 * t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initial condition: u(x, 0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Backward Euler weak form:
    # (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    # => u*v dx + dt*kappa*grad(u).grad(v) dx = u_n*v dx + dt*f*v dx
    
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = u * v * ufl.dx + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_expr_ufl * v * ufl.dx
    
    # Boundary conditions: u = 0 on boundary (exact solution vanishes on boundary of [0,1]^2)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    bcs = [bc]
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create RHS vector using function space
    b_vec = petsc.create_vector(V)
    
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
    
    # Time stepping
    n_steps = int(round(t_end / dt))
    t_current = 0.0
    total_iterations = 0
    
    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current
        
        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)
        
        # Solve
        solver.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on the same grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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
    
    return result


if __name__ == "__main__":
    start = time.time()
    result = solve()
    elapsed = time.time() - start
    
    u_grid = result["u"]
    
    # Compare with exact solution at t_end = 0.2
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-2 * 0.2) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 (RMS) error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Solver info: {result['solver_info']}")
    print(f"u range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"u_exact range: [{np.min(u_exact):.6f}, {np.max(u_exact):.6f}]")
