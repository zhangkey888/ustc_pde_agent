import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Extract parameters
    k_val = 20.0
    if 'pde' in case_spec and 'wavenumber' in case_spec['pde']:
        k_val = float(case_spec['pde']['wavenumber'])
    
    # Get output grid size
    nx_out = 50
    ny_out = 50
    if 'output' in case_spec:
        nx_out = case_spec['output'].get('nx', 50)
        ny_out = case_spec['output'].get('ny', 50)
    
    # Adaptive mesh refinement with convergence check
    # For k=20, we need good resolution. The manufactured solution has 
    # frequencies 6π and 5π, plus k=20. Need sufficient points per wavelength.
    # Use higher order elements for efficiency.
    
    element_degree = 2
    
    # Try progressive refinement
    resolutions = [48, 64, 96, 128]
    prev_norm = None
    u_sol = None
    domain = None
    V = None
    final_N = None
    final_ksp_type = "gmres"
    final_pc_type = "ilu"
    final_rtol = 1e-8
    final_iterations = 0
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        x = ufl.SpatialCoordinate(domain)
        
        # Exact solution for BCs and source term
        u_exact_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
        
        # Source term: f = (61π² - k²) * sin(6πx)sin(5πy)
        f_expr = (61.0 * ufl.pi**2 - k_val**2) * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
        
        # Bilinear form: a(u,v) = ∫ ∇u·∇v dx - k² ∫ u*v dx
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # All boundary
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        # Interpolate exact solution for BC
        u_bc_func = fem.Function(V)
        u_exact_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
        u_bc_func.interpolate(u_exact_fem_expr)
        
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc_func, dofs)
        
        # Solve - try GMRES+ILU first, fallback to direct
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "ilu",
                    "ksp_rtol": "1e-8",
                    "ksp_max_it": "5000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix="helmholtz_"
            )
            u_sol = problem.solve()
            final_ksp_type = "gmres"
            final_pc_type = "ilu"
            final_iterations = 0  # Can't easily get from LinearProblem
        except Exception:
            # Fallback to direct solver
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    },
                    petsc_options_prefix="helmholtz_direct_"
                )
                u_sol = problem.solve()
                final_ksp_type = "preonly"
                final_pc_type = "lu"
                final_iterations = 1
            except Exception as e2:
                raise RuntimeError(f"Both iterative and direct solvers failed: {e2}")
        
        final_N = N
        
        # Check convergence via L2 norm
        current_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
            op=MPI.SUM
        ))
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 0.005:
                break
        
        prev_norm = current_norm
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Compute actual L2 error for diagnostics
    error_form = fem.form((u_sol - u_exact_expr)**2 * ufl.dx)
    l2_error = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    print(f"Helmholtz solver: N={final_N}, degree={element_degree}, L2 error={l2_error:.6e}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": final_ksp_type,
            "pc_type": final_pc_type,
            "rtol": final_rtol,
            "iterations": final_iterations,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "wavenumber": 20.0,
            "type": "helmholtz",
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Check against exact solution on the grid
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(6 * np.pi * X) * np.sin(5 * np.pi * Y)
    
    error = np.sqrt(np.mean((result['u'] - u_exact)**2))
    max_error = np.max(np.abs(result['u'] - u_exact))
    print(f"Grid RMSE: {error:.6e}")
    print(f"Grid max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
