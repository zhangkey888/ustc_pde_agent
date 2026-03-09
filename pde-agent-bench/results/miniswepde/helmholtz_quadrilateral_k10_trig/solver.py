import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Problem parameters
    k_val = 10.0
    
    # Output grid
    nx_out, ny_out = 50, 50
    
    # Adaptive mesh refinement
    resolutions = [32, 64, 128]
    prev_norm = None
    
    u_sol = None
    final_N = None
    final_degree = None
    final_ksp = None
    final_pc = None
    final_rtol = None
    final_iters = 0
    
    for degree in [2, 3]:
        for N in resolutions:
            comm = MPI.COMM_WORLD
            
            # Create quadrilateral mesh on [0,1]x[0,1]
            domain = mesh.create_unit_square(
                comm, N, N, cell_type=mesh.CellType.quadrilateral
            )
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", degree))
            
            # Spatial coordinates
            x = ufl.SpatialCoordinate(domain)
            
            # Exact solution (for BC and source term)
            u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])
            
            # Source term: f = -∇²u - k²u = (13π² - k²) * u_exact
            # ∇²u = -(4π² + 9π²)u = -13π²u
            # -∇²u = 13π²u
            # f = 13π²u - k²u = (13π² - k²)u
            f_expr = (13.0 * ufl.pi**2 - k_val**2) * u_exact_expr
            
            # Trial and test functions
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            
            # Bilinear form: a(u,v) = ∫(∇u·∇v - k²uv)dx
            a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k_val**2 * ufl.inner(u, v)) * ufl.dx
            
            # Linear form: L(v) = ∫fv dx
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
            u_exact_fem_expr = fem.Expression(
                u_exact_expr, V.element.interpolation_points
            )
            u_bc_func.interpolate(u_exact_fem_expr)
            
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            bc = fem.dirichletbc(u_bc_func, dofs)
            
            # Solve - Helmholtz with k=10 is indefinite, use direct solver or GMRES
            ksp_type = "gmres"
            pc_type = "lu"
            rtol = 1e-10
            
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "ksp_rtol": str(rtol),
                        "ksp_max_it": "2000",
                    },
                    petsc_options_prefix="helmholtz_"
                )
                uh = problem.solve()
            except Exception:
                # Fallback to direct solver
                ksp_type = "preonly"
                pc_type = "lu"
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                    },
                    petsc_options_prefix="helmholtz_"
                )
                uh = problem.solve()
            
            # Compute L2 error
            error_form = fem.form(
                ufl.inner(uh - u_exact_expr, uh - u_exact_expr) * ufl.dx
            )
            error_local = fem.assemble_scalar(error_form)
            error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
            
            current_norm = np.sqrt(comm.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)),
                op=MPI.SUM
            ))
            
            u_sol = uh
            final_N = N
            final_degree = degree
            final_ksp = ksp_type
            final_pc = pc_type
            final_rtol = rtol
            final_domain = domain
            
            # Check convergence
            if error_global < 1e-4:
                break
            
            if prev_norm is not None:
                rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
                if rel_change < 1e-5:
                    break
            
            prev_norm = current_norm
        else:
            continue
        break
    
    # Evaluate solution on 50x50 grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points_3d)
    
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": final_degree,
            "ksp_type": final_ksp,
            "pc_type": final_pc,
            "rtol": final_rtol,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {elapsed:.3f}s")
    
    # Compare with exact solution on the grid
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(2*np.pi*XX) * np.cos(3*np.pi*YY)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error vs exact: {error:.6e}")
    print(f"Max error vs exact: {max_error:.6e}")
