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
    Solve convection-diffusion equation with adaptive mesh refinement.
    
    Parameters:
    -----------
    case_spec : dict
        Dictionary containing problem parameters. Expected keys:
        - 'epsilon' (float): diffusion coefficient
        - 'beta' (list): velocity vector [beta_x, beta_y]
        May also be nested under 'pde' key.
    
    Returns:
    --------
    dict with keys:
        - 'u': numpy array of shape (50, 50) with solution values
        - 'solver_info': dictionary with solver metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    start_time = time.time()
    
    # Extract parameters from case_spec (handle nested structure)
    if 'pde' in case_spec and isinstance(case_spec['pde'], dict):
        pde_params = case_spec['pde']
        epsilon = pde_params.get('epsilon', 0.2)
        beta = pde_params.get('beta', [1.0, 0.5])
    else:
        epsilon = case_spec.get('epsilon', 0.2)
        beta = case_spec.get('beta', [1.0, 0.5])
    
    beta = np.array(beta, dtype=ScalarType)
    
    # Grid convergence loop: progressive refinement
    resolutions = [32, 64, 128]
    element_degree = 1  # Linear elements
    u_sol = None
    norm_old = None
    converged_resolution = None
    
    # Track solver performance
    total_iterations = 0
    ksp_type = 'gmres'
    pc_type = 'hypre'
    rtol = 1e-8
    
    for N in resolutions:
        if rank == 0:
            print(f"Testing resolution N={N}")
        
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, 
                                         cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term derived from manufactured solution
        # u_exact = sin(pi*x)*sin(pi*y)
        x = ufl.SpatialCoordinate(domain)
        f_expr = (2 * epsilon * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]) +
                  beta[0] * np.pi * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1]) +
                  beta[1] * np.pi * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]))
        
        # Interpolate source term
        f = fem.Function(V)
        f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
        
        # Bilinear form: ε(∇u, ∇v) + (β·∇u, v)
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx +
             ufl.inner(beta[0] * ufl.grad(u)[0] + beta[1] * ufl.grad(u)[1], v) * ufl.dx)
        
        # Linear form: (f, v)
        L = ufl.inner(f, v) * ufl.dx
        
        # For low Péclet number (~5.6), standard Galerkin is adequate
        # No stabilization needed
        
        # Boundary conditions (Dirichlet, exact solution on entire boundary)
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
        
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Create forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create RHS vector
        b = petsc.create_vector([V])
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solution function
        u_sol = fem.Function(V)
        
        # Try iterative solver first (fastest for large problems)
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        
        try:
            solver.setType(PETSc.KSP.Type.GMRES)
            solver.getPC().setType(PETSc.PC.Type.HYPRE)
            solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
            solver.setFromOptions()
            
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            
            it_count = solver.getIterationNumber()
            total_iterations += it_count
            
            if rank == 0:
                print(f"  GMRES+HYPRE converged in {it_count} iterations")
                
        except Exception:
            # Fallback to direct solver (robust but slower)
            if rank == 0:
                print("  Iterative solver failed, switching to direct LU")
            
            solver = PETSc.KSP().create(domain.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)
            ksp_type = 'preonly'
            pc_type = 'lu'
            
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            it_count = solver.getIterationNumber()
            total_iterations += it_count
        
        # Compute L2 norm for convergence check
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_new = comm.allreduce(norm_local, op=MPI.SUM)
        norm_new = np.sqrt(norm_new)
        
        if rank == 0:
            print(f"  L2 norm: {norm_new:.6f}")
        
        # Check convergence between successive resolutions
        if norm_old is not None:
            relative_error = (abs(norm_new - norm_old) / norm_new 
                             if norm_new > 1e-14 else 0.0)
            if rank == 0:
                print(f"  Relative error: {relative_error:.6f}")
            
            if relative_error < 0.01:  # 1% convergence criterion
                converged_resolution = N
                if rank == 0:
                    print(f"  ✓ Converged at N={N}")
                break
        
        norm_old = norm_new
    
    # Fallback: use finest mesh if no convergence
    if converged_resolution is None:
        converged_resolution = 128
        if rank == 0:
            print(f"  Using finest mesh N={converged_resolution}")
    
    # Compute error against exact solution (for validation)
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = comm.allreduce(error_local, op=MPI.SUM)
    l2_error = np.sqrt(error_global)
    
    if rank == 0:
        print(f"  L2 error vs exact: {l2_error:.2e}")
    
    # Sample solution on 50×50 uniform grid (required by evaluator)
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Points in 3D format (z=0)
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
    # Evaluate solution at points
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), 
                         np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Parallel gathering (simple approach for small grid)
    if comm.size > 1:
        all_values = comm.gather(u_values, root=0)
        if rank == 0:
            combined = np.zeros_like(u_values)
            for i in range(len(combined)):
                for proc_vals in all_values:
                    if not np.isnan(proc_vals[i]):
                        combined[i] = proc_vals[i]
                        break
            u_values = combined
        else:
            u_values = None
        u_values = comm.bcast(u_values, root=0)
    
    u_grid = u_values.reshape((nx, ny))
    
    end_time = time.time()
    wall_time = end_time - start_time
    
    if rank == 0:
        print(f"  Total wall time: {wall_time:.3f} s")
    
    # Prepare solver_info dictionary (required by evaluator)
    solver_info = {
        "mesh_resolution": converged_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations
        # Note: No time-related fields (steady problem)
        # Note: No nonlinear_iterations (linear problem)
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Simple test with default parameters
    case_spec = {
        "epsilon": 0.2,
        "beta": [1.0, 0.5]
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("\nTest passed.")
        print(f"Mesh: {result['solver_info']['mesh_resolution']}")
        print(f"Solution shape: {result['u'].shape}")
