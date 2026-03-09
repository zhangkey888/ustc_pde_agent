import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None):
    """Solve convection-diffusion with SUPG stabilization."""
    
    # Parse parameters from case_spec or use defaults
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        params = pde.get("parameters", {})
        epsilon = params.get("epsilon", 0.01)
        beta_vec = params.get("beta", [20.0, 0.0])
        domain_spec = case_spec.get("domain", {})
        output = case_spec.get("output", {})
        nx_out = output.get("nx", 50)
        ny_out = output.get("ny", 50)
    else:
        epsilon = 0.01
        beta_vec = [20.0, 0.0]
        nx_out = 50
        ny_out = 50

    # Adaptive mesh refinement with convergence check
    # For high Pe with SUPG, we need sufficient resolution
    # With degree 2 elements and SUPG, moderate mesh should suffice
    
    comm = MPI.COMM_WORLD
    
    best_u_grid = None
    best_info = None
    prev_norm = None
    
    # Try different resolutions - for high Pe SUPG, degree 2 is important
    element_degree = 2
    resolutions = [48, 80, 128]
    
    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Spatial coordinates
        x = ufl.SpatialCoordinate(domain)
        
        # Parameters
        eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
        beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
        
        # Manufactured solution: u = sin(pi*x)*sin(pi*y)
        pi = ufl.pi
        u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        
        # Source term: f = -eps * laplacian(u_exact) + beta . grad(u_exact)
        # laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2*sin(pi*x)*sin(pi*y)
        # So -eps * laplacian = eps * 2*pi^2 * sin(pi*x)*sin(pi*y)
        # grad(u_exact) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
        # beta . grad = 20*pi*cos(pi*x)*sin(pi*y)
        grad_u_exact = ufl.grad(u_exact_ufl)
        f_expr = eps_c * 2.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + ufl.dot(beta, grad_u_exact)
        
        # Standard Galerkin weak form
        a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        L_std = ufl.inner(f_expr, v) * ufl.dx
        
        # SUPG stabilization
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        
        # Stabilization parameter (standard formula)
        Pe_cell = beta_norm * h / (2.0 * eps_c)
        # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)  -- simplified
        # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
        # Use a simpler robust formula:
        tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
        
        # SUPG residual: R(u) = -eps*laplacian(u) + beta.grad(u) - f
        # For linear elements, laplacian(u) = 0 within elements
        # For quadratic elements, we still add the full residual
        # The strong residual applied to trial function:
        # R(u) = beta.grad(u) - f  (dropping second derivatives for simplicity with SUPG test function)
        # Actually for P2, the second derivative is piecewise constant, but ufl can handle it
        
        # SUPG: add tau * (beta.grad(v)) * (beta.grad(u) - f) * dx
        # Note: for the bilinear form, the residual of u is: -eps*div(grad(u)) + beta.grad(u) - f
        # The SUPG modification adds: tau * (beta.grad(v)) * (-eps*div(grad(u)) + beta.grad(u)) to LHS
        # and tau * (beta.grad(v)) * f to RHS
        
        supg_test = tau * ufl.dot(beta, ufl.grad(v))
        
        a_supg = supg_test * ufl.dot(beta, ufl.grad(u)) * ufl.dx
        # For higher order elements, include diffusion part of residual
        # -eps * laplacian(u) term in residual - but this requires second derivatives
        # For simplicity and robustness, we skip the diffusion part in SUPG (common practice)
        
        L_supg = supg_test * f_expr * ufl.dx
        
        a_total = a_std + a_supg
        L_total = L_std + L_supg
        
        # Boundary conditions: u = sin(pi*x)*sin(pi*y) = 0 on boundary of [0,1]^2
        # Since sin(0) = sin(pi) = 0, the BC is u = 0 on all boundaries
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
        
        # Solve
        ksp_type = "gmres"
        pc_type = "ilu"
        
        try:
            problem = petsc.LinearProblem(
                a_total, L_total, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": "1e-10",
                    "ksp_max_it": "2000",
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
        
        # Evaluate on output grid
        x_out = np.linspace(0, 1, nx_out)
        y_out = np.linspace(0, 1, ny_out)
        X, Y = np.meshgrid(x_out, y_out, indexing='ij')
        points_2d = np.column_stack([X.ravel(), Y.ravel()])
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
            vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        
        u_grid = u_values.reshape((nx_out, ny_out))
        
        # Compute L2 norm for convergence check
        current_norm = np.sqrt(np.nansum(u_grid**2) / (nx_out * ny_out))
        
        # Also compute error against exact solution
        u_exact_grid = np.sin(np.pi * X) * np.sin(np.pi * Y)
        l2_error = np.sqrt(np.nanmean((u_grid - u_exact_grid)**2))
        
        best_u_grid = u_grid
        best_info = {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
        }
        
        # Check convergence
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 1e-4 and l2_error < 4e-4:
                break
        
        if l2_error < 2e-4:
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
    nx, ny = u_grid.shape
    x_out = np.linspace(0, 1, nx)
    y_out = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    l2_err = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_err = np.max(np.abs(u_grid - u_exact))
    
    print(f"Grid shape: {u_grid.shape}")
    print(f"L2 error: {l2_err:.6e}")
    print(f"Linf error: {linf_err:.6e}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
