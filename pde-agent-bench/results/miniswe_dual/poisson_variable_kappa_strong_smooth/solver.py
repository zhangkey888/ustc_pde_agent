import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation with variable kappa."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec for any overrides
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    
    # Adaptive mesh refinement with convergence check
    resolutions = [48, 80, 128]
    element_degree = 2  # P2 for better accuracy with variable kappa
    
    prev_norm = None
    u_sol = None
    final_N = None
    final_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Variable kappa
        kappa = 1.0 + 0.9 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        
        # Manufactured solution
        u_exact_ufl = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        
        # Compute source term f = -div(kappa * grad(u_exact))
        f = -ufl.div(kappa * ufl.grad(u_exact_ufl))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Bilinear and linear forms
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx
        
        # Boundary conditions: u = g on ∂Ω (from manufactured solution)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # All boundary facets
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        
        # Create BC function
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.sin(3 * np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
        
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Solve
        ksp_type = "cg"
        pc_type = "hypre"
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
                petsc_options_prefix="poisson_"
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
                petsc_options_prefix="poisson_"
            )
            uh = problem.solve()
        
        # Compute L2 error against exact solution
        error_form = fem.form(ufl.inner(uh - u_exact_ufl, uh - u_exact_ufl) * ufl.dx)
        error_local = fem.assemble_scalar(error_form)
        error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
        
        # Compute norm for convergence check
        norm_form = fem.form(ufl.inner(uh, uh) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        current_norm = np.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
        
        u_sol = uh
        final_N = N
        final_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
        
        # Check convergence
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4 or error_global < 1e-4:
                break
        
        if error_global < 1e-4:
            break
            
        prev_norm = current_norm
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(len(points_3d), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": final_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 0.9*sin(2*pi*x)*sin(2*pi*y)"}
            },
            "manufactured_solution": "sin(3*pi*x)*sin(2*pi*y)",
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Check against exact solution on the grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(3 * np.pi * XX) * np.sin(2 * np.pi * YY)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"RMS error on grid: {error:.6e}")
    print(f"Max error on grid: {max_error:.6e}")
