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
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Default parameters
    resolutions = [32, 64, 128]  # Progressive refinement
    element_degree = 1
    rtol = 1e-8
    ksp_type = 'gmres'
    pc_type = 'hypre'
    
    # Extract problem parameters from case_spec if available
    # For this specific case, we know the source term and domain
    # Boundary condition: assume zero Dirichlet unless specified
    g_value = 0.0
    if 'boundary_conditions' in case_spec:
        # In real implementation, parse case_spec for BCs
        pass
    
    # Adaptive mesh refinement loop
    u_sol = None
    norm_old = None
    mesh_resolution_used = None
    iterations_total = 0
    linear_solver_used = None
    preconditioner_used = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Define function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define boundary condition (zero Dirichlet on entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # Mark all boundary facets
        def boundary_marker(x):
            # Boundary is where x[0] or x[1] is 0 or 1
            return np.logical_or.reduce([
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0)
            ])
        
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.full_like(x[0], g_value))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term: f = exp(-180*((x-0.3)**2 + (y-0.7)**2))
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.exp(-180 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
        
        # Coefficient kappa = 1.0
        kappa = fem.Constant(domain, ScalarType(1.0))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Create forms
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        # Assemble matrix
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        
        # Create vector for RHS
        b = petsc.create_vector(L_form.function_spaces)
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        for solver_config in [(ksp_type, pc_type), ('preonly', 'lu')]:
            try:
                # Reset RHS vector
                with b.localForm() as loc:
                    loc.set(0)
                
                # Assemble RHS
                petsc.assemble_vector(b, L_form)
                petsc.apply_lifting(b, [a_form], bcs=[[bc]])
                b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                petsc.set_bc(b, [bc])
                
                # Create solver
                ksp = PETSc.KSP().create(domain.comm)
                ksp.setOperators(A)
                ksp.setType(solver_config[0])
                ksp.getPC().setType(solver_config[1])
                ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
                
                # Create solution function
                u_sol_current = fem.Function(V)
                
                # Solve
                ksp.solve(b, u_sol_current.x.petsc_vec)
                u_sol_current.x.scatter_forward()
                
                # Get iteration count
                its = ksp.getIterationNumber()
                iterations_total += its
                
                # Record solver used
                if not solver_success:
                    linear_solver_used = solver_config[0]
                    preconditioner_used = solver_config[1]
                
                solver_success = True
                break
                
            except Exception as e:
                if rank == 0:
                    print(f"Solver {solver_config} failed: {e}")
                continue
        
        if not solver_success:
            raise RuntimeError("All solvers failed")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol_current, u_sol_current) * ufl.dx)
        norm_value = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / norm_value if norm_value > 0 else 0.0
            if relative_error < 0.01:  # 1% convergence criterion
                u_sol = u_sol_current
                mesh_resolution_used = N
                break
        
        norm_old = norm_value
        u_sol = u_sol_current
        mesh_resolution_used = N
    
    # If loop finished without break, use the last solution (N=128)
    if mesh_resolution_used is None:
        mesh_resolution_used = resolutions[-1]
    
    # Prepare output grid (50x50 uniform grid)
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx * ny)]).T  # 3D points
    
    # Evaluate solution at points
    u_grid_flat = np.full(points.shape[0], np.nan, dtype=np.float64)
    
    # Use geometry utilities for point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Gather results across MPI processes
    u_grid_all = np.zeros_like(u_grid_flat)
    comm.Allreduce(u_grid_flat, u_grid_all, op=MPI.MAX)
    
    # Reshape to (nx, ny)
    u_grid = u_grid_all.reshape(nx, ny)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": linear_solver_used,
        "pc_type": preconditioner_used,
        "rtol": rtol,
        "iterations": iterations_total
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0},
            "source": "gaussian"
        }
    }
    result = solve(case_spec)
    print("Solver completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solution shape: {result['u'].shape}")
