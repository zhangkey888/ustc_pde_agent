import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t_start = time.time()
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    time_params = pde.get("time", {})
    domain_spec = case_spec.get("domain", {})
    
    kappa = coeffs.get("kappa", 0.1)
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    # Agent-selectable parameters - tuned for accuracy within time budget
    N = 128  # mesh resolution
    degree = 2  # polynomial degree
    dt = 0.005  # smaller dt for better temporal accuracy
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # ensure exact t_end
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Time parameter as a Constant
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    # Manufactured solution: u = exp(-0.5*t)*sin(2*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -0.5 * exp(-0.5*t) * sin(2*pi*x) * sin(pi*y)
    # laplacian(u) = exp(-0.5*t) * (-(2*pi)^2 - pi^2) * sin(2*pi*x) * sin(pi*y)
    #              = -5*pi^2 * exp(-0.5*t) * sin(2*pi*x) * sin(pi*y)
    # f = du/dt - kappa * laplacian(u)
    #   = -0.5 * u + kappa * 5*pi^2 * u
    #   = u * (-0.5 + 5*kappa*pi^2)
    f_expr = ufl.exp(-0.5 * t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]) * (-0.5 + 5.0 * kappa * pi**2)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Previous solution
    u_n = fem.Function(V)
    
    # Initialize with exact solution at t=0
    u_n.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f(t_{n+1})
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    
    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_expr * v) * ufl.dx
    
    # Boundary conditions - u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # BC function
    u_bc = fem.Function(V)
    
    # Create expression for exact solution for interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Solution function
    u_sol = fem.Function(V)
    
    # Set up solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    
    total_iterations = 0
    
    # Time stepping
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        
        # Update BC
        u_bc.interpolate(u_exact_expr)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Assemble RHS
        b = petsc.assemble_vector(L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.setOperators(A)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
        
        # Clean up
        A.destroy()
        b.destroy()
    
    solver.destroy()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape((nx_out, ny_out))
    
    # Also evaluate initial condition on same grid
    u_initial_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_initial_grid[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_grid.reshape((nx_out, ny_out))
    
    # Compute error against exact solution at t_end
    u_exact_final = np.exp(-0.5 * t_end) * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    error = np.sqrt(np.mean((u_grid - u_exact_final)**2))
    
    elapsed = time.time() - t_start
    print(f"Mesh: {N}x{N}, degree: {degree}, dt: {dt}, steps: {n_steps}")
    print(f"L2 grid error: {error:.6e}")
    print(f"Total iterations: {total_iterations}")
    print(f"Elapsed time: {elapsed:.2f}s")
    
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
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 0.1},
            "time": {"t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"},
        },
        "domain": {"type": "unit_square"},
    }
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
