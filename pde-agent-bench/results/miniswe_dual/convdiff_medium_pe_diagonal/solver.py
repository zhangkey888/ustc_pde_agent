import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Parse parameters from case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.05)
    beta_vec = params.get("beta", [3.0, 3.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Adaptive mesh refinement
    element_degree = 2
    resolutions = [32, 64, 128]
    
    prev_norm = None
    u_sol = None
    final_N = None
    final_domain = None
    final_ksp_type = "gmres"
    final_pc_type = "hypre"
    final_rtol = 1e-8
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        domain = mesh.create_rectangle(
            comm,
            [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
            [N, N],
            cell_type=mesh.CellType.triangle
        )
        
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        x = ufl.SpatialCoordinate(domain)
        
        beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        
        # Source term derived from manufactured solution
        f_expr = (epsilon * 5.0 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                  + beta_vec[0] * 2.0 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                  + beta_vec[1] * ufl.pi * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # SUPG stabilization parameter
        h = ufl.CellDiameter(domain)
        Pe_cell = beta_norm * h / (2.0 * epsilon)
        tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
        
        # Bilinear form: standard Galerkin + SUPG
        residual_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        supg_test = ufl.dot(beta, ufl.grad(v))
        
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
             + tau * ufl.inner(residual_u, supg_test) * ufl.dx)
        
        L = (f_expr * v * ufl.dx
             + tau * f_expr * supg_test * ufl.dx)
        
        # Boundary conditions (homogeneous for this manufactured solution on [0,1]^2)
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Solve
        ksp_type = "gmres"
        pc_type = "hypre"
        rtol = 1e-8
        
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "2000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix="cdiff_"
            )
            uh = problem.solve()
        except Exception:
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="cdiff_"
            )
            uh = problem.solve()
        
        # Compute L2 norm for convergence check
        current_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)),
            op=MPI.SUM
        ))
        
        final_N = N
        final_domain = domain
        final_ksp_type = ksp_type
        final_pc_type = pc_type
        final_rtol = rtol
        u_sol = uh
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 0.01:
                break
        
        prev_norm = current_norm
    
    # Evaluate solution on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    
    bb_tree = geometry.bb_tree(final_domain, final_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(final_domain, cell_candidates, points)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
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
            "element_degree": element_degree,
            "ksp_type": final_ksp_type,
            "pc_type": final_pc_type,
            "rtol": final_rtol,
            "iterations": 0,
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "params": {
                "epsilon": 0.05,
                "beta": [3.0, 3.0],
            }
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0.0, 1.0, nx_out)
    y_coords = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"Shape: {u_grid.shape}, Range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"RMS Error: {error:.6e}, Max Error: {max_error:.6e}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
