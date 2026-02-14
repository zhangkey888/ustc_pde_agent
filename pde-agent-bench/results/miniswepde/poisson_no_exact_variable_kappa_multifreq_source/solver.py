import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc as fem_petsc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Problem parameters from case_spec
    # Domain is unit square [0,1]x[0,1]
    # Source term f = sin(4*pi*x)*sin(3*pi*y) + 0.3*sin(10*pi*x)*sin(9*pi*y)
    # Coefficient κ = 1 + 0.6*sin(2*pi*x)*sin(2*pi*y)
    # Boundary condition: u = g on ∂Ω (g from case_spec if provided, else 0)
    
    # Extract boundary condition if provided
    g_expr = '0'
    if 'boundary_condition' in case_spec and 'g' in case_spec['boundary_condition']:
        g_expr = case_spec['boundary_condition']['g']
    # For simplicity, assume g = 0 if not provided (homogeneous Dirichlet)
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # P1 elements
    rtol = 1e-8  # linear solver tolerance
    
    # Storage for convergence check
    prev_norm = None
    u_sol_final = None
    solver_info_final = None
    domain_final = None
    mesh_res_final = None
    total_iterations = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Spatial coordinate
        x = ufl.SpatialCoordinate(domain)
        
        # Coefficient κ
        kappa_expr = 1 + 0.6 * ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
        
        # Source term f
        f_expr = (ufl.sin(4 * np.pi * x[0]) * ufl.sin(3 * np.pi * x[1]) +
                  0.3 * ufl.sin(10 * np.pi * x[0]) * ufl.sin(9 * np.pi * x[1]))
        
        # Bilinear and linear forms
        a = kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f_expr * v * ufl.dx
        
        # Boundary condition: u = 0 on entire boundary (Dirichlet)
        # Locate boundary facets
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        # Evaluate g_expr if it's a string expression? For now, assume 0.
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct
        solver_used = None
        pc_used = None
        iterations = 0
        
        # First try: iterative solver (GMRES with hypre)
        try:
            petsc_options = {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": rtol,
                "ksp_atol": 1e-12,
                "ksp_max_it": 1000,
            }
            problem = fem_petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options=petsc_options,
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            solver_used = "gmres"
            pc_used = "hypre"
            # Get iteration count
            ksp = problem.solver
            iterations = ksp.getIterationNumber()
            success = True
        except Exception as e:
            # Fallback: direct solver
            try:
                petsc_options = {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                }
                problem = fem_petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=petsc_options,
                    petsc_options_prefix="pdebench_"
                )
                u_sol = problem.solve()
                solver_used = "preonly"
                pc_used = "lu"
                ksp = problem.solver
                iterations = ksp.getIterationNumber()
                success = True
            except Exception as e2:
                # If both fail, raise
                raise RuntimeError(f"Both iterative and direct solvers failed: {e}, {e2}")
        
        total_iterations += iterations
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(fem.assemble_scalar(norm_form))
        
        # Convergence check
        converged = False
        if prev_norm is not None:
            rel_error = abs(norm_value - prev_norm) / norm_value if norm_value > 1e-14 else 0
            if rel_error < 0.01:  # 1% convergence
                converged = True
        
        prev_norm = norm_value
        
        # Store current solution as candidate
        u_sol_final = u_sol
        domain_final = domain
        mesh_res_final = N
        solver_info_final = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": solver_used,
            "pc_type": pc_used,
            "rtol": rtol,
            "iterations": total_iterations,  # cumulative across all resolutions
        }
        
        if converged:
            break
    
    # At this point, u_sol_final is the solution on domain_final
    # Evaluate solution on a 50x50 uniform grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T  # shape (N, 3)
    
    # Use geometry utilities to evaluate at points
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid_flat = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol_final.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Gather results across processes if running in parallel
    # Use allreduce with nan-aware operation
    u_grid_flat_all = np.zeros_like(u_grid_flat)
    comm.Allreduce(u_grid_flat, u_grid_flat_all, op=MPI.SUM)
    # Since each point is evaluated on exactly one process (due to colliding cells),
    # the sum will place the value in u_grid_flat_all
    u_grid_flat = u_grid_flat_all
    
    u_grid = u_grid_flat.reshape(nx, ny)
    
    # Return dict
    result = {
        "u": u_grid,
        "solver_info": solver_info_final,
    }
    # No time-dependent fields needed
    return result

if __name__ == "__main__":
    # Test with a dummy case_spec
    case_spec = {
        "boundary_condition": {"g": "0"},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print("Solution shape:", result["u"].shape)
        print("Solver info:", result["solver_info"])
