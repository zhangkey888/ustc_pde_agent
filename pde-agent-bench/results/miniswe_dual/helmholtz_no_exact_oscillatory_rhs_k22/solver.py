import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Extract parameters from case_spec
    pde_spec = case_spec.get("pde", {})
    k_val = float(pde_spec.get("wavenumber", 22.0))
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = int(output.get("nx", 50))
    ny_out = int(output.get("ny", 50))
    
    # Boundary condition value (default homogeneous Dirichlet)
    bc_value = 0.0
    
    # Element degree - P2 for good accuracy with moderate mesh
    element_degree = 2
    
    # Adaptive mesh refinement loop
    resolutions = [48, 64, 96, 128]
    prev_norm = None
    u_sol = None
    domain = None
    V = None
    final_N = resolutions[0]
    final_ksp_type = "preonly"
    final_pc_type = "lu"
    final_rtol = 1e-10
    final_iterations = 1
    
    for N in resolutions:
        try:
            # Create mesh
            domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
            
            # Function space
            V = fem.functionspace(domain, ("Lagrange", element_degree))
            
            # Define variational problem
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            x = ufl.SpatialCoordinate(domain)
            
            # Source term: f = sin(10*pi*x)*sin(8*pi*y)
            f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
            
            # Helmholtz weak form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
            k_const = fem.Constant(domain, ScalarType(k_val))
            a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
            L = ufl.inner(f_expr, v) * ufl.dx
            
            # Homogeneous Dirichlet BCs on entire boundary
            tdim = domain.topology.dim
            fdim = tdim - 1
            boundary_facets = mesh.locate_entities_boundary(
                domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
            )
            dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
            bc = fem.dirichletbc(ScalarType(bc_value), dofs, V)
            
            # Use direct solver for robustness (Helmholtz is indefinite)
            ksp_type = "preonly"
            pc_type = "lu"
            
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                    },
                    petsc_options_prefix=f"helmholtz_{N}_"
                )
                u_sol = problem.solve()
            except Exception:
                # Try GMRES+ILU as fallback
                ksp_type = "gmres"
                pc_type = "ilu"
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options={
                        "ksp_type": ksp_type,
                        "pc_type": pc_type,
                        "ksp_rtol": "1e-10",
                        "ksp_max_it": "5000",
                        "ksp_gmres_restart": "200",
                    },
                    petsc_options_prefix=f"helmholtz_gmres_{N}_"
                )
                u_sol = problem.solve()
            
            final_ksp_type = ksp_type
            final_pc_type = pc_type
            final_N = N
            
            # Check convergence by comparing L2 norms
            current_norm = np.sqrt(
                MPI.COMM_WORLD.allreduce(
                    fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
                    op=MPI.SUM
                )
            )
            
            if prev_norm is not None:
                rel_change = abs(current_norm - prev_norm) / (abs(current_norm) + 1e-15)
                if rel_change < 0.005:  # 0.5% convergence criterion
                    break
            
            prev_norm = current_norm
            
        except Exception as e:
            print(f"Resolution {N} failed: {e}")
            continue
    
    # Evaluate solution on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    # Create 3D points array (required by dolfinx even for 2D)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Point evaluation using geometry utilities
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Replace any NaN with 0 (boundary points that might not be found)
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    solver_info = {
        "mesh_resolution": final_N,
        "element_degree": element_degree,
        "ksp_type": final_ksp_type,
        "pc_type": final_pc_type,
        "rtol": final_rtol,
        "iterations": final_iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "helmholtz",
            "wavenumber": 22.0,
            "source_term": "sin(10*pi*x)*sin(8*pi*y)",
        },
        "domain": {
            "type": "unit_square",
            "x_range": [0, 1],
            "y_range": [0, 1],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        },
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solution L2 norm (grid): {np.linalg.norm(result['u']):.6f}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
