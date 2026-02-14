import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    
    Parameters
    ----------
    case_spec : dict
        Dictionary containing problem specification.
        Expected keys: 'pde' with 'type' (e.g., 'elliptic').
        If 'pde' contains 'time', it is ignored for this elliptic problem.
    
    Returns
    -------
    dict
        Contains:
        - "u": numpy array shape (50, 50) with solution values
        - "solver_info": dict with mesh_resolution, element_degree, 
          ksp_type, pc_type, rtol, iterations
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 1
    
    # Store solutions and grid evaluations for convergence check
    u_grids = []      # Solutions evaluated on 50x50 output grid
    
    # Solver info to be populated
    solver_info = {
        "mesh_resolution": None,
        "element_degree": element_degree,
        "ksp_type": None,
        "pc_type": None,
        "rtol": 1e-8,
        "iterations": 0
    }
    
    # Define output grid points (50x50)
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    points = np.zeros((3, nx * ny))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            points[0, idx] = x_vals[i]
            points[1, idx] = y_vals[j]
    
    # Adaptive refinement loop
    for i, N in enumerate(resolutions):
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Source term as UFL expression (from problem description)
        x = ufl.SpatialCoordinate(domain)
        f = ufl.sin(6*ufl.pi*x[0]) * ufl.sin(5*ufl.pi*x[1]) + \
            0.4 * ufl.sin(11*ufl.pi*x[0]) * ufl.sin(9*ufl.pi*x[1])
        
        # Weak form: -div(kappa * grad(u)) = f, with kappa = 1.0
        kappa = 1.0
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Boundary condition: u = 0 on entire boundary (Dirichlet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        def boundary_marker(x):
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
        
        # Try iterative solver first, fallback to direct if fails
        u_sol = fem.Function(V)
        iterations_this_resolution = 0
        
        # First try: iterative solver with GMRES and hypre
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            
            # Get iteration count
            ksp = problem._solver
            iterations_this_resolution = ksp.getIterationNumber()
            
            solver_info["ksp_type"] = "gmres"
            solver_info["pc_type"] = "hypre"
            
        except Exception:
            # Fallback: direct solver with LU
            problem = petsc.LinearProblem(
                a, L, bcs=[bc], u=u_sol,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "ksp_rtol": 1e-8
                },
                petsc_options_prefix="poisson_"
            )
            u_sol = problem.solve()
            
            # For direct solver, iterations = 1 (conceptual)
            iterations_this_resolution = 1
            
            solver_info["ksp_type"] = "preonly"
            solver_info["pc_type"] = "lu"
        
        # Evaluate solution on the 50x50 output grid
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        
        for idx in range(points.shape[1]):
            links = colliding_cells.links(idx)
            if len(links) > 0:
                points_on_proc.append(points.T[idx])
                cells_on_proc.append(links[0])
                eval_map.append(idx)
        
        u_values_all = np.full((points.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values_all[eval_map] = vals.flatten()
        
        u_grid = u_values_all.reshape((ny, nx))
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        
        u_grids.append(u_grid)
        solver_info["iterations"] += iterations_this_resolution
        
        # Check convergence based on change in output grid values
        if i > 0:
            diff = np.abs(u_grids[i] - u_grids[i-1])
            max_diff = diff.max()
            # Tolerance: 1e-4 (conservative relative to accuracy requirement 2.3e-02)
            if max_diff < 1e-4:
                solver_info["mesh_resolution"] = N
                break
        
        # If we reach the last resolution, use it
        if i == len(resolutions) - 1:
            solver_info["mesh_resolution"] = N
    
    # Final solution is the last one in the list (either converged or finest)
    final_u_grid = u_grids[-1]
    
    return {
        "u": final_u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Test the solver with a dummy case_spec
    case_spec = {
        "pde": {
            "type": "elliptic"
        }
    }
    result = solve(case_spec)
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
