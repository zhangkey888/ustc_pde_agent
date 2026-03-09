import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization.
    
    -ε ∇²u + β·∇u = f   in Ω
      u = g               on ∂Ω
    
    Manufactured solution: u = sin(πx)sin(πy)
    """
    
    # Default parameters
    epsilon = 0.01
    beta_vec = [10.0, 10.0]
    nx_out = 50
    ny_out = 50
    
    # Parse case_spec if provided
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        coeffs = pde.get("coefficients", {})
        
        # Try multiple possible keys for epsilon
        epsilon = float(params.get("epsilon", coeffs.get("epsilon", epsilon)))
        beta_vec = params.get("beta", coeffs.get("beta", beta_vec))
        beta_vec = [float(b) for b in beta_vec]
        
        output = case_spec.get("output", {})
        nx_out = output.get("nx", nx_out)
        ny_out = output.get("ny", ny_out)
    
    comm = MPI.COMM_WORLD
    
    # Adaptive mesh refinement with convergence check
    # For high Pe with SUPG + degree 2, moderate mesh should suffice
    configs = [(64, 2), (96, 2), (128, 2)]
    
    best_u_grid = None
    best_info = None
    prev_error = None
    
    for N, deg in configs:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", deg))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Exact solution in UFL
        u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        
        # Source term: f = -ε ∇²u + β·∇u
        # u = sin(πx)sin(πy)
        # ∇u = (π cos(πx)sin(πy), π sin(πx)cos(πy))
        # ∇²u = -2π² sin(πx)sin(πy)
        # -ε ∇²u = 2επ² sin(πx)sin(πy)
        # β·∇u = β₁ π cos(πx)sin(πy) + β₂ π sin(πx)cos(πy)
        f_expr = (2.0 * epsilon * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                  + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                  + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Velocity vector
        beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
        
        # Standard Galerkin bilinear form and RHS
        a_std = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                 + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx)
        L_std = f_expr * v * ufl.dx
        
        # SUPG stabilization
        h = ufl.CellDiameter(domain)
        beta_mag = ufl.sqrt(ufl.dot(beta, beta))
        
        # Element Peclet number
        Pe_h = beta_mag * h / (2.0 * epsilon)
        
        # SUPG parameter tau
        # Using the standard formula with coth approximation
        # For high Pe: tau ≈ h/(2|β|)
        # For low Pe: tau ≈ h²/(12ε)
        # Smooth transition:
        tau = h / (2.0 * beta_mag) * (1.0 - 1.0 / Pe_h)
        # Clamp for safety (when Pe_h could be small)
        tau = ufl.conditional(ufl.gt(Pe_h, 1.0),
                              h / (2.0 * beta_mag) * (1.0 - 1.0 / Pe_h),
                              h * h / (12.0 * epsilon))
        
        # SUPG stabilization term
        # Test function modification: tau * β·∇v
        # Adds: ∫ tau * (β·∇v) * (-ε∇²u + β·∇u) dx
        # For the bilinear form (operator on u):
        #   ∫ tau * (β·∇v) * (β·∇u) dx  (convection part)
        #   - ε * ∫ tau * (β·∇v) * ∇²u dx  (diffusion part, often skipped for CG)
        # For CG elements, ∇²u is not well-defined element-wise for linear elements
        # but for degree >= 2 it can be included. We'll skip it for robustness.
        
        supg_test = tau * ufl.dot(beta, ufl.grad(v))
        
        a_supg = ufl.dot(beta, ufl.grad(u)) * supg_test * ufl.dx
        L_supg = f_expr * supg_test * ufl.dx
        
        # Total forms
        a_total = a_std + a_supg
        L_total = L_std + L_supg
        
        # Boundary conditions
        # u = sin(πx)sin(πy) = 0 on ∂Ω of [0,1]²
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool))
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
        
        # Solve with GMRES (non-symmetric system due to convection)
        ksp_type = "gmres"
        pc_type = "ilu"
        
        try:
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": "1e-10",
                    "ksp_atol": "1e-14",
                    "ksp_max_it": "3000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix="cdiff_"
            )
            u_sol = problem.solve()
        except Exception:
            # Fallback to direct solver
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix="cdiff_"
            )
            u_sol = problem.solve()
        
        # Compute L2 error against exact solution
        u_exact_func = fem.Function(V)
        expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_exact_func.interpolate(expr)
        
        error_form = fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx)
        error_local = fem.assemble_scalar(error_form)
        l2_error = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
        
        # Evaluate on output grid
        xs = np.linspace(0, 1, nx_out)
        ys = np.linspace(0, 1, ny_out)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        points_2d = np.column_stack([XX.ravel(), YY.ravel()])
        points_3d = np.zeros((points_2d.shape[0], 3))
        points_3d[:, :2] = points_2d
        
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
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape((nx_out, ny_out))
        
        best_u_grid = u_grid
        best_info = {
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
            "l2_error": float(l2_error),
        }
        
        # Check if converged
        if l2_error < 3e-4:
            break
        
        prev_error = l2_error
    
    return {
        "u": best_u_grid,
        "solver_info": best_info,
    }


if __name__ == "__main__":
    import time
    
    # Test with the convection-diffusion case
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.01,
                "beta": [10.0, 10.0],
            },
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"L2 error: {result['solver_info']['l2_error']:.6e}")
    print(f"Mesh: {result['solver_info']['mesh_resolution']}, Degree: {result['solver_info']['element_degree']}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    
    # Also compute pointwise error on the grid
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    rms_error = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact_grid))
    print(f"Grid RMS error: {rms_error:.6e}")
    print(f"Grid Max error: {max_error:.6e}")
