import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None):
    """
    Solve convection-diffusion equation:
      -eps * laplacian(u) + beta . grad(u) = f  in Omega
      u = g on dOmega
    with SUPG stabilization for high Peclet number.
    Manufactured solution: u = sin(2*pi*x)*sin(2*pi*y)
    """
    # Parse parameters (with defaults matching the problem)
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    params = pde_spec.get("params", {})
    
    epsilon = params.get("epsilon", 0.02)
    beta_vec = params.get("beta", [6.0, 2.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output_spec = case_spec.get("output", {})
    nx_out = output_spec.get("nx", 50)
    ny_out = output_spec.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    
    # Adaptive mesh refinement with convergence check
    # For high Peclet, we need SUPG and possibly fine mesh
    # Try degree 2 with SUPG for better accuracy
    element_degree = 2
    
    # We'll use a convergence loop
    resolutions = [64, 96, 128]
    prev_norm = None
    best_u_grid = None
    best_info = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_rectangle(
            comm,
            [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
            [N, N],
            cell_type=mesh.CellType.triangle
        )
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Exact solution (manufactured)
        u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        
        # Compute source term f from manufactured solution
        # -eps * laplacian(u_exact) + beta . grad(u_exact) = f
        # u_exact = sin(2*pi*x)*sin(2*pi*y)
        # laplacian(u_exact) = -8*pi^2 * sin(2*pi*x)*sin(2*pi*y)
        # So -eps * laplacian = eps * 8*pi^2 * sin(2*pi*x)*sin(2*pi*y)
        # grad(u_exact) = [2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y)]
        # beta . grad = beta[0]*2*pi*cos(2*pi*x)*sin(2*pi*y) + beta[1]*2*pi*sin(2*pi*x)*cos(2*pi*y)
        
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        beta = fem.Constant(domain, PETSc.ScalarType(np.array(beta_vec, dtype=np.float64)))
        
        # Source term via UFL (automatic differentiation)
        grad_u_exact = ufl.grad(u_exact_expr)
        laplacian_u_exact = ufl.div(ufl.grad(u_exact_expr))
        f_expr = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Standard Galerkin weak form
        a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # SUPG stabilization
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        Pe_cell = beta_norm * h / (2.0 * eps_c)
        
        # SUPG stabilization parameter (tau)
        # Using the standard formula with coth-based optimal tau
        tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
        
        # SUPG residual: strong form residual tested with tau * beta . grad(v)
        # Strong residual for trial function: -eps*laplacian(u) + beta.grad(u) - f
        # For linear problem, we split into bilinear and linear parts
        # The SUPG test function modification: v_supg = tau * beta . grad(v)
        v_supg = tau * ufl.dot(beta, ufl.grad(v))
        
        # SUPG addition to bilinear form:
        # For the strong operator applied to u: -eps*div(grad(u)) + beta.grad(u)
        # Since u is piecewise polynomial of degree 2, div(grad(u)) is nonzero for degree >= 2
        # But for simplicity with TrialFunction, we use the consistent SUPG form
        a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
        # For degree 2, the Laplacian of u is piecewise constant (nonzero)
        # But computing -eps * div(grad(u)) for TrialFunction in standard FE is tricky
        # We'll add the diffusion part only if using DG or high-order
        # For CG, the second derivative is not well-defined across elements
        # Standard SUPG only adds the convection part to the test function
        
        L_supg = ufl.inner(f_expr, v_supg) * ufl.dx
        
        a_total = a + a_supg
        L_total = L + L_supg
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        # All boundary
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        
        # Interpolate exact solution for BC
        u_bc = fem.Function(V)
        u_exact_interp = fem.Expression(u_exact_expr, V.element.interpolation_points)
        u_bc.interpolate(u_exact_interp)
        
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Solve
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        
        try:
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "2000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix="convdiff_"
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
                petsc_options_prefix="convdiff_"
            )
            u_sol = problem.solve()
        
        # Evaluate on output grid
        x_coords = np.linspace(x_range[0], x_range[1], nx_out)
        y_coords = np.linspace(y_range[0], y_range[1], ny_out)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        points = np.zeros((3, nx_out * ny_out))
        points[0, :] = X.flatten()
        points[1, :] = Y.flatten()
        
        # Point evaluation
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points[:, i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        
        u_values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            pts_arr = np.array(points_on_proc)
            cells_arr = np.array(cells_on_proc, dtype=np.int32)
            vals = u_sol.eval(pts_arr, cells_arr)
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape((nx_out, ny_out))
        
        # Compute L2 norm for convergence check
        current_norm = np.sqrt(np.nansum(u_grid**2))
        
        # Also compute error against exact solution
        u_exact_vals = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        error = np.sqrt(np.nanmean((u_grid - u_exact_vals)**2))
        
        best_u_grid = u_grid
        best_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,  # Not easily accessible from LinearProblem
        }
        
        # Check convergence
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4 or error < 1e-3:
                break
        
        if error < 1e-3:
            break
            
        prev_norm = current_norm
    
    return {
        "u": best_u_grid,
        "solver_info": best_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    # Compute error against exact solution
    nx_out, ny_out = u_grid.shape
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    
    l2_error = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Mesh resolution: {info['mesh_resolution']}")
    print(f"Element degree: {info['element_degree']}")
    print(f"L2 error: {l2_error:.6e}")
    print(f"Linf error: {linf_error:.6e}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Target error: <= 3.08e-03")
    print(f"Target time: <= 2.471s")
    print(f"PASS: {l2_error <= 3.08e-3 and elapsed <= 2.471}")
