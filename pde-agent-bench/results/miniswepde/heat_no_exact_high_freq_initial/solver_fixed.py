import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters from case_spec with defaults
    t_end = case_spec.get('pde', {}).get('time', {}).get('t_end', 0.12)
    dt_suggested = case_spec.get('pde', {}).get('time', {}).get('dt', 0.02)
    scheme = case_spec.get('pde', {}).get('time', {}).get('scheme', 'backward_euler')
    
    # Force is_transient = True since we have time parameters
    is_transient = True
    
    # Grid convergence loop parameters
    resolutions = [32, 64, 128]
    element_degree = 1  # P1 elements for efficiency
    converged = False
    u_final = None
    mesh_resolution_used = None
    solver_info = {}
    
    # Time stepping parameters
    dt = dt_suggested
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps  # Adjust dt to exactly reach t_end
    
    # Track solver iterations
    total_linear_iterations = 0
    
    # Adaptive mesh refinement loop
    norm_old = None
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (Dirichlet, u = 0 on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
            # Mark entire boundary
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define coefficients
        kappa = ScalarType(1.0)  # thermal diffusivity
        
        # Define initial condition u0 = sin(6*pi*x)*sin(6*pi*y)
        u_n = fem.Function(V)
        def u0_expr(x):
            return np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1])
        u_n.interpolate(u0_expr)
        
        # Time-stepping loop
        u_current = fem.Function(V)
        u_current.x.array[:] = u_n.x.array
        
        # Define forms for time-dependent problem
        # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = 0
        a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix (constant in time for linear problem)
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vectors
        b = petsc.create_vector(L_form.function_spaces)
        u_sol = fem.Function(V)
        
        # Create linear solver with iterative first, fallback to direct
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)
            solver.setFromOptions()
            
            # Time stepping
            for step in range(n_steps):
                # Assemble RHS
                with b.localForm() as loc:
                    loc.set(0)
                petsc.assemble_vector(b, L_form)
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Solve linear system
                solver.solve(b, u_sol.x.petsc_vec)
                u_sol.x.scatter_forward()
                
                # Get iteration count
                its = solver.getIterationNumber()
                total_linear_iterations += its
                
                # Update for next step
                u_n.x.array[:] = u_sol.x.array
            
            solver_converged = True
            ksp_type = "gmres"
            pc_type = "hypre"
            
        except Exception as e:
            # Fallback to direct solver
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            
            # Reset u_n to initial condition
            u_n.interpolate(u0_expr)
            
            # Time stepping with direct solver
            for step in range(n_steps):
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
                
                # Get iteration count (direct solver typically 1 iteration)
                its = solver.getIterationNumber()
                total_linear_iterations += its
                
                # Update for next step
                u_n.x.array[:] = u_sol.x.array
            
            solver_converged = True
            ksp_type = "preonly"
            pc_type = "lu"
        
        # Compute L2 norm of final solution
        norm_new = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)), op=MPI.SUM))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_new - norm_old) / norm_new if norm_new > 0 else 0
            if relative_error < 0.01:  # 1% convergence criterion
                converged = True
                u_final = u_sol
                mesh_resolution_used = N
                break
        
        norm_old = norm_new
        u_final = u_sol
        mesh_resolution_used = N
    
    # If loop finished without convergence, use finest mesh result
    if not converged:
        u_final = u_sol
        mesh_resolution_used = resolutions[-1]
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Combine results from all processes
    # Each process has u_values with NaN for points not found locally
    # We need to combine so that non-NaN values from any process are used
    u_values_combined = np.full_like(u_values, np.nan)
    comm.Allreduce(u_values, u_values_combined, op=MPI.MAX)
    # MPI.MAX will ignore NaN? Actually NaN > number is False, so NaN will be ignored if any process has a valid number
    # But if all processes have NaN for a point, it remains NaN (should not happen)
    
    # For safety, also check if any points still NaN and fill with 0
    nan_mask = np.isnan(u_values_combined)
    if np.any(nan_mask):
        # This shouldn't happen if mesh covers domain
        u_values_combined[nan_mask] = 0.0
    
    u_grid = u_values_combined.reshape(ny, nx)
    
    # Also compute initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    
    u0_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_values_combined = np.full_like(u0_values, np.nan)
    comm.Allreduce(u0_values, u0_values_combined, op=MPI.MAX)
    nan_mask0 = np.isnan(u0_values_combined)
    if np.any(nan_mask0):
        u0_values_combined[nan_mask0] = 0.0
    
    u_initial = u0_values_combined.reshape(ny, nx)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a sample case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.12,
                "dt": 0.02,
                "scheme": "backward_euler"
            }
        }
    }
    
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Time steps: {result['solver_info']['n_steps']}")
