import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation: -ε ∇²u + β·∇u = f
    with SUPG stabilization for high Péclet numbers.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem parameters:
        - epsilon: diffusion coefficient (float)
        - beta: convection velocity vector (list of floats)
        
    Returns:
    --------
    dict with keys:
        - "u": numpy array of shape (50, 50) with solution values
        - "solver_info": dictionary with solver metadata
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters from case_spec with defaults
    epsilon = case_spec.get('epsilon', 0.05)
    beta = case_spec.get('beta', [3.0, 3.0])
    beta_array = np.array(beta, dtype=ScalarType)
    
    # Manufactured solution: u = sin(2πx) * sin(πy)
    def exact_solution(x):
        return np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1])
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    solutions = []  # Store (solution, domain, iterations)
    errors = []     # Store L2 errors
    
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space with linear elements
        degree = 1
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Convection velocity as UFL constant
        beta_const = fem.Constant(domain, beta_array)
        
        # Define source term f from manufactured solution
        # f = -ε ∇²u_exact + β·∇u_exact
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        f_ufl = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_const, ufl.grad(u_exact_ufl))
        
        # SUPG stabilization parameter (Brooks-Hughes formula)
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta_const, beta_const))
        tau = h / (2 * beta_norm) * (1 / ufl.tanh(beta_norm * h / (2 * epsilon)) - 
                                    2 * epsilon / (beta_norm * h))
        
        # Variational form (bilinear part)
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + 
             ufl.inner(ufl.dot(beta_const, ufl.grad(u)), v)) * ufl.dx
        
        # Add SUPG stabilization to bilinear form
        a += tau * ufl.inner(ufl.dot(beta_const, ufl.grad(u)), ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Linear form
        L = ufl.inner(f_ufl, v) * ufl.dx
        
        # Add SUPG stabilization to linear form
        L += tau * ufl.inner(f_ufl, ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Dirichlet boundary conditions (all boundaries)
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Boundary function from exact solution
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Create forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble stiffness matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create and assemble RHS vector
        b = petsc.create_vector(V)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solution function
        u_sol = fem.Function(V)
        
        # Setup linear solver (GMRES with HYPRE preconditioner)
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.GMRES)
        solver.getPC().setType(PETSc.PC.Type.HYPRE)
        solver.setTolerances(rtol=1e-8)
        
        # Solve linear system
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Get iteration count
        iterations = solver.getIterationNumber()
        
        # Compute L2 error against exact solution
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: exact_solution(x))
        error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
        error_value = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
        
        errors.append(error_value)
        solutions.append((u_sol, domain, iterations))
        
        # Check convergence: stop if error change < 1%
        if i > 0:
            relative_error = abs(errors[i] - errors[i-1]) / errors[i] if errors[i] != 0 else 0
            if relative_error < 0.01:
                if comm.rank == 0:
                    print(f"Converged at resolution N={N} with L2 error {error_value:.6e}")
                break
    
    # Use the finest available solution
    u_final, domain_final, final_iterations = solutions[-1]
    N_final = resolutions[min(len(solutions)-1, len(resolutions)-1)]
    final_error = errors[-1]
    
    if comm.rank == 0:
        print(f"Final mesh resolution: {N_final}")
        print(f"Final L2 error: {final_error:.6e} (target: ≤ 2.40e-03)")
    
    # Sample solution on 50×50 uniform grid
    nx, ny = 50, 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten for point evaluation (need 3D coordinates)
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).astype(ScalarType)
    
    # Evaluate solution at grid points
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather values from all MPI processes
    u_values_all = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_all, op=MPI.MAX)
    
    # Reshape to 50×50 grid
    u_grid = u_values_all.reshape((nx, ny))
    
    # Prepare solver metadata
    solver_info = {
        "mesh_resolution": N_final,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": final_iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test with the given convection-diffusion case
    case_spec = {
        "epsilon": 0.05,
        "beta": [3.0, 3.0]
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Solution shape: {result['u'].shape}")
        print(f"Solver info: {result['solver_info']}")
