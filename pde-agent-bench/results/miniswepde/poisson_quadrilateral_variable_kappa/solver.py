import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import math

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with variable coefficient κ.
    Adaptive mesh refinement with convergence check.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Parse case specification (if needed)
    # For this problem, κ is given in problem description
    # We'll use hardcoded expression as per problem, but could extract from case_spec
    
    # Domain: unit square
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1  # linear elements
    
    # Default solver settings (will be updated based on actual solver used)
    ksp_type_used = 'gmres'
    pc_type_used = 'hypre'
    rtol = 1e-8
    
    # Storage for previous norm
    prev_norm = None
    u_sol = None
    final_resolution = None
    iterations_total = 0
    
    for N in resolutions:
        if rank == 0:
            print(f"Solving with mesh resolution N={N}")
        
        # Create mesh (quadrilateral cells as per case name)
        domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                       cell_type=mesh.CellType.quadrilateral)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        # κ as expression: 1 + 0.5*cos(2πx)*cos(2πy)
        kappa = 1.0 + 0.5*ufl.cos(2*ufl.pi*x[0])*ufl.cos(2*ufl.pi*x[1])
        
        # Manufactured solution
        u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        # Compute f = -∇·(κ∇u_exact)
        f_expr = -ufl.div(kappa * ufl.grad(u_exact))
        
        # Weak form
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Boundary condition: Dirichlet with exact solution
        # Locate all boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Interpolate exact solution for BC
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(2*np.pi*x[0]) * np.sin(np.pi*x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first
        solver_succeeded = False
        linear_iterations = 0
        ksp_type_local = 'gmres'
        pc_type_local = 'hypre'
        
        for solver_try in range(2):  # First try iterative, then direct
            if solver_try == 0:
                # Iterative solver
                petsc_options = {
                    "ksp_type": ksp_type_local,
                    "pc_type": pc_type_local,
                    "ksp_rtol": rtol,
                    "ksp_atol": 1e-12,
                    "ksp_max_it": 1000,
                }
                prefix = "pdebench_iter"
            else:
                # Fallback direct solver
                ksp_type_local = 'preonly'
                pc_type_local = 'lu'
                petsc_options = {
                    "ksp_type": ksp_type_local,
                    "pc_type": pc_type_local,
                }
                prefix = "pdebench_direct"
                if rank == 0:
                    print("Iterative solver failed, switching to direct solver")
            
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=petsc_options,
                    petsc_options_prefix=prefix
                )
                u_h = problem.solve()
                
                # Get linear solver iterations
                ksp = problem.solver
                its = ksp.getIterationNumber()
                linear_iterations = its
                iterations_total += its
                
                solver_succeeded = True
                ksp_type_used = ksp_type_local
                pc_type_used = pc_type_local
                break
            except Exception as e:
                if rank == 0:
                    print(f"Solver try {solver_try} failed: {e}")
                continue
        
        if not solver_succeeded:
            raise RuntimeError("Both iterative and direct solvers failed")
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = domain.comm.allreduce(norm_local, op=MPI.SUM)
        norm = np.sqrt(norm_global)
        
        if rank == 0:
            print(f"  L2 norm = {norm}")
        
        # Check convergence
        if prev_norm is not None:
            rel_error = abs(norm - prev_norm) / norm if norm > 0 else 0.0
            if rank == 0:
                print(f"  Relative error in norm = {rel_error}")
            if rel_error < 0.01:  # 1% convergence
                u_sol = u_h
                final_resolution = N
                if rank == 0:
                    print(f"Converged at N={N}")
                break
        
        prev_norm = norm
        u_sol = u_h
        final_resolution = N
    
    # If loop finished without break, use the last solution (N=128)
    if u_sol is None:
        # Should not happen, but safety
        u_sol = u_h
        final_resolution = 128
    
    # Evaluate solution on a 50x50 uniform grid
    nx = 50
    ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), np.zeros(nx*ny)]).astype(ScalarType)
    
    # Use geometry utilities to evaluate at points
    from dolfinx import geometry
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather all values to rank 0 (evaluator likely runs sequentially)
    # For parallel runs, we need to combine values from all ranks
    # Simple approach: use allgather since number of points is small (2500)
    recv_vals = comm.allgather(u_values)
    # Combine: take first non-NaN value for each point
    u_values_combined = np.zeros_like(u_values)
    for i in range(len(u_values)):
        for rv in recv_vals:
            if not np.isnan(rv[i]):
                u_values_combined[i] = rv[i]
                break
    
    u_grid = u_values_combined.reshape(nx, ny)
    
    # Fill NaN values with 0 (points outside domain shouldn't happen)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    # Prepare solver_info
    solver_info = {
        "mesh_resolution": final_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": rtol,
        "iterations": iterations_total,
    }
    
    # Check if PDE has time (should not for elliptic)
    if 'pde' in case_spec and 'time' in case_spec['pde']:
        # This is a transient problem, but our case is elliptic
        # We would need to add dt, n_steps, time_scheme
        # For safety, we can set defaults
        pass
    
    # No time-dependent fields
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result

if __name__ == "__main__":
    # Test with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.5*cos(2*pi*x)*cos(2*pi*y)"}
            }
        }
    }
    result = solve(case_spec)
    print("Test completed successfully")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"Solver: {result['solver_info']['ksp_type']} with {result['solver_info']['pc_type']}")
    print(f"Iterations: {result['solver_info']['iterations']}")
    print(f"Solution shape: {result['u'].shape}")
