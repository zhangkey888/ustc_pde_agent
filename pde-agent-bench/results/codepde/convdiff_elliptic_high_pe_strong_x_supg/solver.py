import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.01)
    beta_vec = pde_config.get("beta", [15.0, 0.0])
    
    # Mesh resolution - use higher resolution for high Peclet number
    N = 80
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Spatial coordinates and exact solution
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = sin(pi*x)*sin(pi*y)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -eps * laplacian(u) + beta . grad(u)
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So -eps * laplacian = eps * 2 * pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad(u) = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(np.array(beta_vec, dtype=np.float64)))
    
    f_expr = (eps_val * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_vec[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # Stabilization parameter (standard formula)
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ~ h/(2*|beta|) for large Pe
    # Simplified: for high Peclet
    tau = h / (2.0 * beta_norm + 1e-10)
    
    # SUPG residual: L_strong(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For trial function (linear), laplacian may be zero for P1 but not for P2
    # We use the strong form residual
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    
    # SUPG test function modification
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = a_standard + ufl.inner(r_u, v_supg) * ufl.dx
    # Note: r_u contains f_expr which goes to L side
    # Let's split properly:
    # a_supg has the u-dependent part of r_u * v_supg
    # L_supg has the f-dependent part
    
    # Actually, let's rewrite more carefully:
    # r_u = (-eps*div(grad(u)) + beta.grad(u)) - f
    # So: integral of r_u * v_supg = integral of (-eps*div(grad(u)) + beta.grad(u)) * v_supg - f * v_supg
    
    a_form = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
              + tau * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dot(beta, ufl.grad(v)) * ufl.dx)
    
    L_form = (f_expr * v * ufl.dx
              + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx)
    
    # 6. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # u = sin(pi*x)*sin(pi*y) = 0 on all boundaries of unit square
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 7. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 8. Extract solution on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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