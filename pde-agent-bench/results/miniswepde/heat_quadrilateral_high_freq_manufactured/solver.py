import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the transient heat equation with adaptive mesh refinement.
    """
    start_time = time.time()
    
    # Extract parameters with defaults from Problem Description
    t_end = 0.1
    dt = 0.005
    time_scheme = "backward_euler"
    
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Manufactured solution
    def exact_solution(x, t):
        return np.exp(-t) * np.sin(4*np.pi*x[0]) * np.sin(4*np.pi*x[1])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    comm = MPI.COMM_WORLD
    element_degree = 1
    
    prev_norm = None
    u_final = None
    mesh_resolution_used = None
    u_initial = None
    
    # For solver info
    ksp_type_used = 'gmres'
    pc_type_used = 'hypre'
    rtol_used = 1e-8
    total_iterations = 0
    
    for N in resolutions:
        print(f"Testing mesh resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary condition setup - mark all boundaries
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary condition function (initial time)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x, 0.0))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Coefficients
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        # Functions for time-stepping
        u_n = fem.Function(V)  # Previous time step
        u_np1 = fem.Function(V)  # Current time step
        
        # Initial condition
        u_n.interpolate(lambda x: exact_solution(x, 0.0))
        
        if u_initial is None:
            u_initial = u_n.copy()
        
        # Time-stepping parameters
        n_steps = int(np.ceil(t_end / dt))
        dt_adjusted = t_end / n_steps
        
        # Spatial coordinate for source term
        x = ufl.SpatialCoordinate(domain)
        
        # Backward Euler forms
        a = (1/dt_adjusted) * u * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        a_form = fem.form(a)
        
        # Assemble matrix with BCs applied (matrix is constant)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Try iterative solver first
        solver_success = False
        ksp_type = 'gmres'
        pc_type = 'hypre'
        rtol = 1e-8
        iterations_this_mesh = 0
        
        for solver_try in range(2):
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            
            if solver_try == 0:
                # Iterative solver
                solver.setType(PETSc.KSP.Type.GMRES)
                solver.getPC().setType(PETSc.PC.Type.HYPRE)
                solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                ksp_type = 'gmres'
                pc_type = 'hypre'
            else:
                # Direct solver fallback
                solver.setType(PETSc.KSP.Type.PREONLY)
                solver.getPC().setType(PETSc.PC.Type.LU)
                ksp_type = 'preonly'
                pc_type = 'lu'
            
            try:
                # Time stepping
                for step in range(n_steps):
                    t = (step + 1) * dt_adjusted
                    
                    # Update boundary condition values for current time
                    u_bc.interpolate(lambda x: exact_solution(x, t))
                    # bc object already has correct dofs, just update the function
                    
                    # Source term at time t
                    f_expr = (32*np.pi**2 - 1) * ufl.exp(-t) * ufl.sin(4*np.pi*x[0]) * ufl.sin(4*np.pi*x[1])
                    
                    # Linear form
                    L = (1/dt_adjusted) * u_n * v * ufl.dx + f_expr * v * ufl.dx
                    L_form = fem.form(L)
                    
                    # Assemble RHS
                    b = petsc.assemble_vector(L_form)
                    
                    # Apply boundary conditions to RHS
                    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    petsc.set_bc(b, [bc])
                    
                    # Solve
                    solver.solve(b, u_np1.x.petsc_vec)
                    u_np1.x.scatter_forward()
                    
                    # Get iteration count
                    its = solver.getIterationNumber()
                    iterations_this_mesh += its
                    
                    # Update for next step
                    u_n.x.array[:] = u_np1.x.array
                    
                    # Clean up RHS vector
                    b.destroy()
                
                solver_success = True
                ksp_type_used = ksp_type
                pc_type_used = pc_type
                rtol_used = rtol
                total_iterations += iterations_this_mesh
                break
                
            except Exception as e:
                if solver_try == 0:
                    print(f"  Iterative solver failed, trying direct: {e}")
                    continue
                else:
                    print(f"  Direct solver also failed: {e}")
                    raise
        
        if not solver_success:
            raise RuntimeError(f"All solvers failed for N={N}")
        
        # Compute L2 norm at final time
        norm_form = fem.form(ufl.inner(u_n, u_n) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        print(f"  Final time norm: {norm_value:.6f}")
        
        # Convergence check
        if prev_norm is not None:
            relative_error = abs(norm_value - prev_norm) / norm_value if norm_value > 0 else 1.0
            print(f"  Relative error vs previous mesh: {relative_error:.6f}")
            if relative_error < 0.01:
                print(f"  CONVERGED at N={N}")
                u_final = u_n
                mesh_resolution_used = N
                break
        
        prev_norm = norm_value
        u_final = u_n
        mesh_resolution_used = N
    
    # Use finest mesh if loop finished without convergence
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
    
    # Prepare output on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solutions
    u_grid_flat = evaluate_function_at_points(u_final, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    u_initial_flat = evaluate_function_at_points(u_initial, points)
    u_initial_grid = u_initial_flat.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol_used,
        "iterations": total_iterations,
        "dt": dt_adjusted,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    end_time = time.time()
    print(f"Total solve time: {end_time - start_time:.2f}s, final mesh N={mesh_resolution_used}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }

def evaluate_function_at_points(u_func, points):
    """Evaluate Function at points."""
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
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
    test_spec = {
        "pde": {
            "time": {
                "t_end": 0.1,
                "dt": 0.005,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(test_spec)
    print(f"\nSolution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
