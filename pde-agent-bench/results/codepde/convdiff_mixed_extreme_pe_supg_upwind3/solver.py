import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.002)
    beta_vec = pde_config.get("beta", [25.0, 10.0])
    
    # High Peclet number => need SUPG stabilization and fine mesh
    # Pe ~ |beta| * h / (2*eps), so we need good resolution
    # Use higher-order elements for better accuracy
    
    N = 128  # mesh resolution
    degree = 2  # quadratic elements
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # so -eps * (-2*pi^2 * sin(pi*x)*sin(pi*y)) = 2*eps*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad(u) = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    pi = ufl.pi
    beta = ufl.as_vector([default_scalar_type(beta_vec[0]), default_scalar_type(beta_vec[1])])
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    
    f_expr = (2.0 * eps_val * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vec[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vec[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin bilinear form
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Use the formula: tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ~ 1, so tau ~ h/(2*|beta|) * (1 - 1/Pe)
    # Simpler: tau = h / (2 * |beta|) for high Pe
    # Or use the "optimal" formula
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    # SUPG residual: L_strong(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For trial function (linear), laplacian is available through div(grad(u))
    # But for linear trial functions with P1, laplacian is zero element-wise
    # For P2, we can use it but it's tricky with trial functions
    # Instead, we add SUPG as: tau * (beta . grad(v)) * (beta . grad(u) - f) * dx
    # (dropping the diffusion part of the strong residual in the test function weighting)
    
    r_supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = r_supg_test * ufl.dot(beta, ufl.grad(u)) * ufl.dx
    L_supg = r_supg_test * f_expr * ufl.dx
    
    # For P2 elements, we can also include the diffusion part
    # But div(grad(u)) for P2 on triangles is piecewise constant per element
    # Let's include it for better accuracy
    # a_supg += tau * dot(beta, grad(v)) * (-eps * div(grad(u))) * dx
    # This requires second derivatives of trial function
    # For Lagrange P2 on triangles, div(grad(u)) is well-defined (constant per cell)
    a_supg_diff = r_supg_test * (-eps_c * ufl.div(ufl.grad(u))) * ufl.dx
    
    a_total = a_standard + a_supg + a_supg_diff
    L_total = L_standard + L_supg
    
    # Boundary conditions: u = sin(pi*x)*sin(pi*y) = 0 on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
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
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "200",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
            "stabilization": "SUPG",
        }
    }