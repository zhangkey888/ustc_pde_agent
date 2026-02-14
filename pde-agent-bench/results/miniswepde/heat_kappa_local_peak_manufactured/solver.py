import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    
    # Extract problem parameters
    # Use hardcoded defaults if not provided (as per instructions)
    t_end = 0.1
    dt = 0.01
    time_scheme = 'backward_euler'
    
    # Check if time parameters are provided in case_spec
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 2  # Use quadratic elements for better accuracy
    solutions = []
    norms = []
    total_linear_iterations = 0
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition (Dirichlet) - use exact solution
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)  # All boundaries
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Time-stepping setup
        u_n = fem.Function(V)  # Previous time step
        u = fem.Function(V)    # Current time step
        
        # Create kappa as a Function (interpolated)
        kappa_func = fem.Function(V)
        def kappa_expr(x):
            return 1.0 + 30.0 * np.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
        kappa_func.interpolate(kappa_expr)
        
        # For source term computation
        def f_func(x, t):
            # f = ∂u/∂t - ∇·(κ ∇u)
            # u = exp(-t)*sin(pi*x)*sin(2*pi*y)
            # ∂u/∂t = -exp(-t)*sin(pi*x)*sin(2*pi*y)
            x_coord = x[0]
            y_coord = x[1]
            pi = np.pi
            
            # Compute κ
            kappa = 1.0 + 30.0 * np.exp(-150.0 * ((x_coord - 0.35)**2 + (y_coord - 0.65)**2))
            
            # Compute ∇κ
            dkappa_dx = 30.0 * np.exp(-150.0 * ((x_coord - 0.35)**2 + (y_coord - 0.65)**2)) * (-300.0 * (x_coord - 0.35))
            dkappa_dy = 30.0 * np.exp(-150.0 * ((x_coord - 0.35)**2 + (y_coord - 0.65)**2)) * (-300.0 * (y_coord - 0.65))
            
            # Compute ∇u
            du_dx = np.exp(-t) * pi * np.cos(pi * x_coord) * np.sin(2 * pi * y_coord)
            du_dy = np.exp(-t) * 2 * pi * np.sin(pi * x_coord) * np.cos(2 * pi * y_coord)
            
            # Compute Δu
            d2u_dx2 = -np.exp(-t) * pi**2 * np.sin(pi * x_coord) * np.sin(2 * pi * y_coord)
            d2u_dy2 = -np.exp(-t) * (2 * pi)**2 * np.sin(pi * x_coord) * np.sin(2 * pi * y_coord)
            
            # ∇·(κ ∇u) = κ Δu + ∇κ · ∇u
            div_kappa_grad_u = kappa * (d2u_dx2 + d2u_dy2) + dkappa_dx * du_dx + dkappa_dy * du_dy
            
            # ∂u/∂t
            du_dt = -np.exp(-t) * np.sin(pi * x_coord) * np.sin(2 * pi * y_coord)
            
            # f = ∂u/∂t - ∇·(κ ∇u)
            f = du_dt - div_kappa_grad_u
            return f
        
        # Initial condition
        def u0_func(x):
            return np.exp(0) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
        
        u_n.interpolate(u0_func)
        u.interpolate(u0_func)
        
        # Define variational problem
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        
        # Time-stepping scheme (Backward Euler)
        # a(u, v) = (u * v + dt * kappa * dot(grad(u), grad(v))) * dx
        # L(v) = (u_n + dt * f) * v * dx
        
        # Bilinear form (left-hand side)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa_func * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        
        # Linear form (right-hand side, will be updated each time step)
        # We'll create this inside the time loop
        
        # Assemble forms
        a_form = fem.form(a)
        
        # Time-stepping loop
        n_steps = int(t_end / dt)
        linear_iterations = 0
        
        # Create function for source term
        f_func_obj = fem.Function(V)
        
        for step in range(n_steps):
            t = (step + 1) * dt
            
            # Update boundary condition with exact solution at current time
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: np.exp(-t) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Update source term
            f_func_obj.interpolate(lambda x: f_func(x, t))
            
            # Create linear form for this time step: L(v) = (u_n + dt * f) * v * dx
            L = ufl.inner(u_n + dt * f_func_obj, v) * ufl.dx
            L_form = fem.form(L)
            
            # Assemble matrix with boundary conditions
            A = petsc.assemble_matrix(a_form, bcs=[bc])
            A.assemble()
            
            # Create vectors
            b = petsc.create_vector(L_form.function_spaces)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply boundary conditions to RHS
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            
            # Create linear solver with iterative first, fallback to direct
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            
            # Try iterative solver first
            try:
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=1e-8, max_it=1000)
                solver.setFromOptions()
            except Exception:
                # Fallback to direct solver
                solver.setType(PETSc.KSP.Type.PREONLY)
                solver.getPC().setType(PETSc.PC.Type.LU)
            
            # Solve
            solver.solve(b, u.x.petsc_vec)
            linear_iterations += solver.getIterationNumber()
            
            # Update solution
            u.x.scatter_forward()
            
            # Prepare for next step
            u_n.x.array[:] = u.x.array[:]
        
        total_linear_iterations += linear_iterations
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u, u) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        norms.append(norm_value)
        
        # Store solution for this resolution
        solutions.append((N, u.copy()))
        
        # Check convergence (compare with previous resolution if available)
        if i > 0:
            relative_error = abs(norms[i] - norms[i-1]) / norms[i] if norms[i] != 0 else 1.0
            if relative_error < 0.01:  # 1% convergence criterion
                print(f"Converged at resolution N={N} with relative error {relative_error:.6f}")
                break
    
    # Use the last solution (either converged or finest mesh)
    final_N, final_u = solutions[-1]
    
    # Prepare output on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points for evaluation (3D coordinates)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Evaluate solution at points
    u_values = evaluate_function_at_points(final_u, points)
    u_grid = u_values.reshape(nx, ny)
    
    # Evaluate initial condition at points
    u0 = fem.Function(V)
    u0.interpolate(lambda x: np.exp(0) * np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
    u0_values = evaluate_function_at_points(u0, points)
    u_initial = u0_values.reshape(nx, ny)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at given points.
    points: shape (3, N) numpy array
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    return u_values

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
