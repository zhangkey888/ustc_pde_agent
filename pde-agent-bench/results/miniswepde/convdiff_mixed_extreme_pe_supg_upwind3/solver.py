import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None):
    """Solve convection-diffusion with SUPG stabilization."""
    
    # Parse parameters
    if case_spec is None:
        case_spec = {}
    
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.002)
    beta_vec = params.get("beta", [25.0, 10.0])
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Manufactured solution: u = sin(pi*x)*sin(pi*y)
    # Source term: f = -eps * laplacian(u) + beta . grad(u)
    # laplacian(u) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # So: -eps * (-2*pi^2 * sin(pi*x)*sin(pi*y)) + beta_x * pi*cos(pi*x)*sin(pi*y) + beta_y * sin(pi*x)*pi*cos(pi*y)
    # f = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta_x*pi*cos(pi*x)*sin(pi*y) + beta_y*pi*sin(pi*x)*cos(pi*y)
    
    # Use degree 2 elements for better accuracy with SUPG
    element_degree = 2
    N = 80  # mesh resolution - balance accuracy vs speed
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Parameters
    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    
    # Source term (derived from manufactured solution)
    pi_ = ufl.pi
    u_exact_ufl = ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1])
    
    # f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u_exact) = [pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    f_expr = (2.0 * epsilon * pi_**2 * ufl.sin(pi_ * x[0]) * ufl.sin(pi_ * x[1])
              + beta_vec[0] * pi_ * ufl.cos(pi_ * x[0]) * ufl.sin(pi_ * x[1])
              + beta_vec[1] * pi_ * ufl.sin(pi_ * x[0]) * ufl.cos(pi_ * x[1]))
    
    # Standard Galerkin terms
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    # Residual operator applied to trial function: -eps*laplacian(u) + beta.grad(u)
    # For linear elements, laplacian of u_h = 0 within elements
    # For quadratic elements, we still include the full residual
    
    # Element size
    h = ufl.CellDiameter(domain)
    
    # Local Peclet number
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_local = beta_mag * h / (2.0 * eps_c)
    
    # SUPG stabilization parameter (optimal for quad/tri elements)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) - 1/Pe ≈ 1
    # Simplified: tau = h / (2 * |beta|) for high Pe
    # More robust formula:
    tau = h / (2.0 * beta_mag + 1e-10)
    
    # SUPG test function modification: v_supg = beta . grad(v)
    # The strong-form residual of the PDE applied to u_h:
    # R(u) = -eps * div(grad(u)) + beta . grad(u) - f
    
    # For the bilinear form, the SUPG contribution is:
    # a_supg = tau * (beta.grad(u)) * (beta.grad(v)) dx  [dominant term for high Pe]
    # Plus: tau * (-eps * laplacian(u)) * (beta.grad(v)) dx [small for high Pe]
    
    # Full SUPG: tau * (-eps*div(grad(u)) + beta.grad(u)) * (beta.grad(v)) * dx
    # RHS SUPG: tau * f * (beta.grad(v)) * dx
    
    beta_grad_v = ufl.dot(beta, ufl.grad(v))
    beta_grad_u = ufl.dot(beta, ufl.grad(u))
    
    # For P2 elements, we can include the diffusion term in the residual
    # but div(grad(u)) for P2 on triangles is piecewise constant
    # Let's include it for completeness
    a_supg = tau * (beta_grad_u - eps_c * ufl.div(ufl.grad(u))) * beta_grad_v * ufl.dx
    L_supg = tau * f_expr * beta_grad_v * ufl.dx
    
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # Boundary conditions: u = sin(pi*x)*sin(pi*y) on boundary
    # On the unit square boundary, sin(pi*x)*sin(pi*y) = 0 everywhere
    # (since either x=0,1 or y=0,1, making sin(pi*x)=0 or sin(pi*y)=0)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": "1e-10",
            "ksp_atol": "1e-12",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Evaluate on output grid
    x_out = np.linspace(x_range[0], x_range[1], nx_out)
    y_out = np.linspace(y_range[0], y_range[1], ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    # Point evaluation
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {
                "epsilon": 0.002,
                "beta": [25.0, 10.0],
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
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # L2 error on grid
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    linf_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"L2 grid error: {error:.6e}")
    print(f"Linf grid error: {linf_error:.6e}")
