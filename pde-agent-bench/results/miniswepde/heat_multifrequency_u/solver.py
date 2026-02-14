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
    Solve the transient heat equation with adaptive mesh refinement.
    """
    # Start timing
    start_time = time.time()
    
    # Extract parameters from case_spec with defaults
    t_end = 0.1
    dt = 0.01
    time_scheme = "backward_euler"
    
    # Override with case_spec if provided
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        time_params = case_spec['pde']['time']
        t_end = time_params.get('t_end', t_end)
        dt = time_params.get('dt', dt)
        time_scheme = time_params.get('scheme', time_scheme)
    
    # Manufactured solution
    def exact_solution(x, t):
        """u_exact = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(6*pi*y))"""
        return np.exp(-t) * (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 
                             0.2 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1]))
    
    # Progressive mesh refinement
    resolutions = [32, 64, 128]
    element_degree = 1
    
    # Variables to store results
    u_final = None
    norm_old = None
    mesh_resolution_used = None
    total_linear_iterations = 0
    solver_info = {}
    
    comm = MPI.COMM_WORLD
    
    # Track solver type
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Define constants
        kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
        
        # Time-stepping setup
        n_steps = int(t_end / dt)
        if n_steps == 0:
            n_steps = 1
            dt = t_end
        
        # Create functions for solution
        u_n = fem.Function(V)  # u at t_n
        u_n1 = fem.Function(V)  # u at t_{n+1}
        
        # Set initial condition
        def u0_expr(x):
            return exact_solution(x, 0.0)
        u_n.interpolate(u0_expr)
        u_n1.x.array[:] = u_n.x.array[:]
        
        # Define boundary condition (Dirichlet: u = g on ∂Ω)
        # g is the exact solution at current time
        t_bc = fem.Constant(domain, PETSc.ScalarType(0.0))  # Time for BC
        
        def boundary_marker(x):
            # Mark entire boundary
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        # Find boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        
        # Locate DOFs on boundary
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create BC function that evaluates exact solution at current time
        u_bc_expr = fem.Expression(
            ufl.exp(-t_bc) * (ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[0]) * 
                             ufl.sin(np.pi * ufl.SpatialCoordinate(domain)[1]) + 
                             0.2 * ufl.sin(6 * np.pi * ufl.SpatialCoordinate(domain)[0]) * 
                             ufl.sin(6 * np.pi * ufl.SpatialCoordinate(domain)[1])),
            V.element.interpolation_points
        )
        u_bc = fem.Function(V)
        
        # Create Dirichlet BC
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        # Define variational problem for backward Euler
        x = ufl.SpatialCoordinate(domain)
        t_var = fem.Constant(domain, PETSc.ScalarType(0.0))  # Time variable for source
        
        # Exact solution as UFL expression
        u_exact_ufl = ufl.exp(-t_var) * (ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) + 
                                         0.2 * ufl.sin(6 * np.pi * x[0]) * ufl.sin(6 * np.pi * x[1]))
        
        # Compute f = ∂u/∂t - Δu
        u_t = -u_exact_ufl
        delta_u = ufl.div(ufl.grad(u_exact_ufl))
        f_ufl = u_t - delta_u
        
        # Create fem.Expression for f
        f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
        f = fem.Function(V)
        
        # Define forms
        a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx
        
        # Assemble forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble stiffness matrix with BCs
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create RHS vector
        b = petsc.create_vector(V)
        
        # Create linear solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        # Try iterative solver first
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=1e-8, max_it=1000)
            solver.setFromOptions()
            # Test solve with zero RHS to check if solver works
            test_vec = b.duplicate()
            solver.solve(test_vec, test_vec)
            ksp_type_used = "gmres"
            pc_type_used = "hypre"
        except Exception:
            # Fallback to direct solver
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.LU)
            ksp_type_used = "preonly"
            pc_type_used = "lu"
        
        # Time-stepping loop
        linear_iterations = 0
        
        for step in range(n_steps):
            # Update time
            t_var.value += dt
            t_bc.value = t_var.value  # Update BC time
            
            # Update f at current time
            f.interpolate(f_expr)
            
            # Update BC function
            u_bc.interpolate(u_bc_expr)
            
            # Assemble RHS
            with b.localForm() as loc:
                loc.set(0)
            petsc.assemble_vector(b, L_form)
            
            # Apply lifting for BCs
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            
            # Apply BCs to RHS
            petsc.set_bc(b, [bc])
            
            # Solve linear system
            solver.solve(b, u_n1.x.petsc_vec)
            u_n1.x.scatter_forward()
            
            # Get iteration count
            linear_iterations += solver.getIterationNumber()
            
            # Update u_n for next step
            u_n.x.array[:] = u_n1.x.array[:]
        
        total_linear_iterations += linear_iterations
        
        # Compute norm of solution for convergence check
        norm_new = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_n1, u_n1) * ufl.dx)))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_new - norm_old) / norm_new if norm_new > 0 else 0
            if relative_error < 0.01:  # 1% convergence
                u_final = u_n1
                mesh_resolution_used = N
                break
        
        norm_old = norm_new
        u_final = u_n1
        mesh_resolution_used = N
    
    # If loop finished without break, use the last result
    if u_final is None:
        u_final = u_n1
        mesh_resolution_used = 128
    
    # Prepare output grid (50x50 uniform grid)
    nx = ny = 50
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Flatten for point evaluation
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T
    
    # Evaluate solution at points
    u_values = evaluate_function_at_points(u_final, points)
    u_grid = u_values.reshape((nx, ny))
    
    # Evaluate initial condition at points
    u0_func = fem.Function(V)
    u0_func.interpolate(u0_expr)
    u0_values = evaluate_function_at_points(u0_func, points)
    u_initial = u0_values.reshape((nx, ny))
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": 1e-8,
        "iterations": total_linear_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme
    }
    
    # End timing
    end_time = time.time()
    wall_time = end_time - start_time
    
    # Print diagnostics
    if comm.rank == 0:
        print(f"Mesh resolution: {mesh_resolution_used}")
        print(f"Total linear iterations: {total_linear_iterations}")
        print(f"Solver: {ksp_type_used}/{pc_type_used}")
        print(f"Wall time: {wall_time:.2f}s")
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }


def evaluate_function_at_points(u_func, points):
    """
    Evaluate a dolfinx Function at an array of points.
    points: shape (N, 3) numpy array
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells colliding with points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Build per-point mapping
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan, dtype=PETSc.ScalarType)
    
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # In parallel, we need to gather results from all processes
    comm = domain.comm
    u_values_all = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_all, op=MPI.SUM)
    
    # Replace any remaining NaN with 0 (points not found on any process)
    u_values_all[np.isnan(u_values_all)] = 0.0
    
    return u_values_all


if __name__ == "__main__":
    # Test the solver with a minimal case_spec
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
    print("Test completed successfully")
    print(f"u shape: {result['u'].shape}")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
