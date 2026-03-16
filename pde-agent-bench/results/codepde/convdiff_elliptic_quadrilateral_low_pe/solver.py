import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.25)
    beta_val = pde_config.get("beta", [1.0, 0.5])
    
    nx_out = case_spec.get("nx", 50)
    ny_out = case_spec.get("ny", 50)
    
    # 2. Create mesh - use quadrilateral as specified
    N = 80
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: sin(pi*x)*sin(pi*y)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Diffusion coefficient and velocity
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    beta = ufl.as_vector([default_scalar_type(beta_val[0]), default_scalar_type(beta_val[1])])
    
    # Source term from manufactured solution:
    # f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian(sin(pi*x)*sin(pi*y)) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So -eps * laplacian = eps * 2*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u_exact) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    f_expr = (eps_val * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_val[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_val[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    
    # Bilinear form: eps * (grad u, grad v) + (beta . grad u) * v
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization for Pe ~ 4.5 (moderate, but stabilization helps)
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG residual applied to test function
    r = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    # SUPG: add tau * (beta . grad v) * r
    a += tau * ufl.dot(beta, ufl.grad(v)) * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx
    
    # 5. Boundary conditions: u = sin(pi*x)*sin(pi*y) = 0 on boundary of unit square
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="cdsolve_"
    )
    uh = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # 7. Extract solution on uniform grid
    xv = np.linspace(0.0, 1.0, nx_out)
    yv = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xv, yv, indexing='ij')
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
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
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
        }
    }