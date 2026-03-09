import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion equation with SUPG stabilization."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [0.0, 15.0])
    
    domain_spec = case_spec.get("domain", {})
    nx_out = domain_spec.get("nx", 50)
    ny_out = domain_spec.get("ny", 50)
    
    # Adaptive mesh refinement
    comm = MPI.COMM_WORLD
    
    # For high Pe, we need good resolution. Start with moderate and go up.
    # With SUPG and degree 2, we can get good accuracy
    element_degree = 2
    N = 80  # mesh resolution
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for BCs and source term
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Diffusion and convection
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    beta = fem.Constant(domain, PETSc.ScalarType(np.array(beta_vec, dtype=np.float64)))
    
    # Source term: f = -eps * laplacian(u) + beta . grad(u)
    # laplacian(u_exact) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # So -eps * laplacian = eps * 2*pi^2 * sin(pi*x)*sin(pi*y)
    # beta . grad(u) = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    f_expr = (epsilon * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin weak form
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Classical formula: tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Simplified: tau = h / (2 * |beta|) for high Pe
    # More robust formula:
    tau = h / (2.0 * beta_norm + 1e-10) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    # SUPG residual: R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements
    # For quadratic elements, we need the full residual
    # The strong residual applied to trial function:
    # -eps * div(grad(u)) + beta . grad(u) - f
    # Note: for P2, div(grad(u)) is not zero but is piecewise constant
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    # SUPG terms
    a_supg = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    # But for the diffusion part in SUPG, we should use the strong form residual
    # Actually, the proper SUPG formulation adds:
    # sum_K integral_K tau * (beta.grad(u) - eps*laplacian(u) - f) * (beta.grad(v)) dx
    # Since laplacian of u (trial) involves second derivatives which are tricky,
    # for P2 elements we can include it, but a simpler approach that works well:
    # Just use: tau * (beta.grad(u) - f) * beta.grad(v) dx (dropping the diffusion term in residual)
    # This is common practice for convection-dominated problems
    
    # Simpler SUPG: only convection part in residual
    a_total = a_std + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_total = L_std + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
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
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.01,
                "beta": [0.0, 15.0],
            }
        },
        "domain": {
            "type": "unit_square",
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
