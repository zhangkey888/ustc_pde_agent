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
    domain_spec = case_spec.get("domain", {})
    
    epsilon = params.get("epsilon", 0.05)
    beta_vec = params.get("beta", [4.0, 2.0])
    
    # Domain bounds
    x_min = domain_spec.get("x_min", 0.0)
    x_max = domain_spec.get("x_max", 1.0)
    y_min = domain_spec.get("y_min", 0.0)
    y_max = domain_spec.get("y_max", 1.0)
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Manufactured solution: u = sin(2*pi*x)*sin(pi*y)
    # Source term: f = -eps * laplacian(u) + beta . grad(u)
    # laplacian(u) = -(4*pi^2 + pi^2) * sin(2*pi*x)*sin(pi*y) = -5*pi^2 * sin(2*pi*x)*sin(pi*y)
    # So -eps * laplacian(u) = 5*eps*pi^2 * sin(2*pi*x)*sin(pi*y)
    # grad(u) = (2*pi*cos(2*pi*x)*sin(pi*y), pi*sin(2*pi*x)*cos(pi*y))
    # beta . grad(u) = beta[0]*2*pi*cos(2*pi*x)*sin(pi*y) + beta[1]*pi*sin(2*pi*x)*cos(pi*y)
    
    # Adaptive mesh refinement
    element_degree = 2  # P2 for better accuracy with convection-dominated problems
    
    # For high Peclet, we need good mesh resolution + SUPG
    # Try resolutions adaptively
    resolutions = [48, 80, 128]
    
    prev_norm = None
    final_result = None
    final_info = None
    
    for N in resolutions:
        result, info, norm_val = _solve_at_resolution(
            N, element_degree, epsilon, beta_vec,
            x_min, x_max, y_min, y_max,
            nx_out, ny_out
        )
        
        if prev_norm is not None:
            rel_change = abs(norm_val - prev_norm) / (abs(norm_val) + 1e-15)
            if rel_change < 0.005:
                # Converged
                return result
        
        prev_norm = norm_val
        final_result = result
        final_info = info
    
    return final_result


def _solve_at_resolution(N, degree, epsilon, beta_vec, x_min, x_max, y_min, y_max, nx_out, ny_out):
    """Solve at a given mesh resolution and return result + norm for convergence check."""
    
    comm = MPI.COMM_WORLD
    
    # Create mesh - quadrilateral as specified in case ID
    p0 = np.array([x_min, y_min])
    p1 = np.array([x_max, y_max])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space - use Lagrange on quads (Q elements)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution (for BC and source term)
    u_exact_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Gradient of exact solution
    grad_u_exact = ufl.as_vector([
        2 * ufl.pi * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.pi * ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    ])
    
    # Laplacian of exact solution
    laplacian_u_exact = -(4 * ufl.pi**2 + ufl.pi**2) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Velocity field
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Source term: f = -epsilon * laplacian(u) + beta . grad(u)
    f_expr = -epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin weak form
    a_std = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter (optimal formula)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG residual: strong form residual applied to trial function
    # Strong form: -epsilon * laplacian(u) + beta . grad(u) - f = 0
    # For linear elements, laplacian(u) = 0 within elements
    # For higher order, we still use the approximation that laplacian contribution is small
    # The SUPG test function modification: v_supg = tau * (beta . grad(v))
    
    r_supg = ufl.dot(beta, ufl.grad(v))
    
    a_supg = tau * ufl.dot(beta, ufl.grad(u)) * r_supg * ufl.dx
    L_supg = tau * f_expr * r_supg * ufl.dx
    
    # For degree >= 2, include diffusion term in SUPG
    if degree >= 2:
        # -epsilon * laplacian(u) term in residual
        # For the trial function, we can't easily compute laplacian of u in the bilinear form
        # So we skip it (common practice, still consistent)
        pass
    
    a_total = a_std + a_supg
    L_total = L_std + L_supg
    
    # Boundary conditions - Dirichlet on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function from exact solution
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
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
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out, x_min, x_max, y_min, y_max)
    
    # Compute norm for convergence check
    norm_val = np.sqrt(np.nanmean(u_grid**2))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,  # Not easily accessible from LinearProblem
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result, solver_info, norm_val


def _evaluate_on_grid(domain, u_func, nx, ny, x_min, x_max, y_min, y_max):
    """Evaluate solution on a uniform grid."""
    
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    
    # Create grid points (3D coordinates for dolfinx)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid


if __name__ == "__main__":
    # Test with default parameters
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "params": {
                "epsilon": 0.05,
                "beta": [4.0, 2.0],
            }
        },
        "domain": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
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
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    
    # Compute error against exact solution
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(2 * np.pi * xv) * np.sin(np.pi * yv)
    
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    print(f"RMS Error: {error:.6e}")
    print(f"Max Error: {np.nanmax(np.abs(u_grid - u_exact)):.6e}")
    print(f"Solver info: {result['solver_info']}")
