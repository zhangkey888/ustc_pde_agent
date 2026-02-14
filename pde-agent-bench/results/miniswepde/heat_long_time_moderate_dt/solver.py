import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from dolfinx import nls
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve the heat equation with adaptive mesh refinement and time-stepping.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters from case_spec with defaults
    # Problem Description says t_end=0.2, dt=0.01, scheme=backward_euler
    # Force is_transient = True as per instructions
    t_end = 0.2
    dt = 0.01
    time_scheme = 'backward_euler'
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Manufactured solution
    def u_exact(x, t):
        return np.exp(-2*t) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
    
    # Source term f = du/dt - κ∇²u
    κ = 0.5  # given coefficient
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    u_sol = None
    u_norm_prev = None
    converged_resolution = None
    solver_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", 1))
        
        # Boundary condition: Dirichlet from exact solution
        # We'll create time-dependent BC in the time loop
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Time-stepping setup
        u = fem.Function(V)
        u_n = fem.Function(V)  # previous time step
        
        # Initial condition
        u_n.interpolate(lambda x: u_exact(x, 0.0))
        u.x.array[:] = u_n.x.array[:]
        
        # Trial and test functions
        v = ufl.TestFunction(V)
        u_trial = ufl.TrialFunction(V)
        
        # Time-stepping parameters
        n_steps = int(np.round(t_end / dt))
        actual_dt = t_end / n_steps  # ensure exact t_end
        
        # Spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Variational form for backward Euler:
        # (u - u_n)/dt * v dx + κ * dot(grad(u), grad(v)) dx = f * v dx
        # where f = du/dt - κ∇²u from manufactured solution
        
        # Left-hand side matrix (constant)
        a = ufl.inner(u_trial, v) * ufl.dx + actual_dt * κ * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        a_form = fem.form(a)
        
        # Assemble matrix (without BCs)
        A = petsc.assemble_matrix(a_form, bcs=[])
        A.assemble()
        
        # Create solver
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        
        # Try iterative solver first
        ksp_type = "gmres"
        pc_type = "hypre"
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.setTolerances(rtol=1e-8, max_it=1000)
        
        # Create vectors
        b = petsc.create_vector([V])
        u_vec = u.x.petsc_vec
        
        # Time stepping
        total_linear_iterations = 0
        t = 0.0
        
        for step in range(n_steps):
            t_prev = t
            t = (step + 1) * actual_dt
            
            # Source term f at time t
            # f = du/dt - κ∇²u = exp(-2*t)*sin(pi*x)*sin(pi*y)*(-2 + π²) for κ=0.5
            f_coeff = (-2.0 + np.pi**2) * np.exp(-2*t)  # κ=0.5, so 2κπ² = π²
            f_expr = f_coeff * ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])
            
            # Right-hand side: (u_n + dt*f) * v dx
            L = ufl.inner(u_n + actual_dt * f_expr, v) * ufl.dx
            L_form = fem.form(L)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Apply boundary conditions (time-dependent)
            # Create BC function with exact solution at current time
            u_bc = fem.Function(V)
            u_bc.interpolate(lambda x: u_exact(x, t))
            bc = fem.dirichletbc(u_bc, dofs)
            
            # Apply lifting to RHS (modifies b for non-homogeneous BCs)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Set BC values in RHS
            petsc.set_bc(b, [bc])
            
            # For matrix, we need to apply BCs too
            # Create a copy of A and zero rows/columns for Dirichlet DOFs
            A_bc = A.copy()
            try:
                # Get PETSc indices for Dirichlet DOFs
                if len(dofs) > 0:
                    # Convert to PETSc indices (0-based)
                    petsc_indices = PETSc.IS().createGeneral(dofs, comm=comm)
                    A_bc.zeroRowsColumns(petsc_indices, 1.0)
                    petsc_indices.destroy()
                
                # Update solver operator
                ksp.setOperators(A_bc)
                
                # Solve linear system
                try:
                    ksp.solve(b, u_vec)
                    it_count = ksp.getIterationNumber()
                    total_linear_iterations += it_count
                except PETSc.Error as e:
                    # Fallback to direct solver
                    ksp_type = "preonly"
                    pc_type = "lu"
                    ksp.setType(PETSc.KSP.Type.PREONLY)
                    ksp.getPC().setType(PETSc.PC.Type.LU)
                    ksp.solve(b, u_vec)
                    it_count = ksp.getIterationNumber()
                    total_linear_iterations += it_count
                
                # Update u_n for next step
                u_n.x.array[:] = u.x.array[:]
                
            finally:
                # Clean up A_bc
                A_bc.destroy()
        
        # Compute norm of solution at final time
        u_norm = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx)), op=MPI.SUM))
        
        # Check convergence
        if u_norm_prev is not None:
            relative_error = abs(u_norm - u_norm_prev) / u_norm if u_norm > 1e-12 else abs(u_norm - u_norm_prev)
            if relative_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                u_sol = u
                # Store solver info
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": 1,
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "rtol": 1e-8,
                    "iterations": total_linear_iterations,
                    "dt": actual_dt,
                    "n_steps": n_steps,
                    "time_scheme": time_scheme
                }
                break
        
        u_norm_prev = u_norm
        u_sol = u  # keep latest solution
        
        # Store solver info for this resolution (will be overwritten if converged earlier)
        solver_info = {
            "mesh_resolution": N,
            "element_degree": 1,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": total_linear_iterations,
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme
        }
    
    # If loop finished without convergence, use finest mesh result
    if converged_resolution is None:
        converged_resolution = 128
    
    # Sample solution on 50x50 grid
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T
    
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
    
    u_values = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # All-gather to ensure all points are filled (in parallel)
    u_values_all = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_all, op=MPI.MAX)
    u_grid = u_values_all.reshape(nx, ny)
    
    # Also compute initial condition on same grid
    u0_func = fem.Function(V)
    u0_func.interpolate(lambda x: u_exact(x, 0.0))
    
    u0_values = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals0 = u0_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u0_values[eval_map] = vals0.flatten()
    
    u0_values_all = np.zeros_like(u0_values)
    comm.Allreduce(u0_values, u0_values_all, op=MPI.MAX)
    u_initial = u0_values_all.reshape(nx, ny)
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a minimal case_spec
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.2,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)
    print("u_initial shape:", result["u_initial"].shape)
