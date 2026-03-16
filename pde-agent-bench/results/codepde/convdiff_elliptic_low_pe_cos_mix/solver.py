import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    eps_val = pde_config.get("epsilon", 0.2)
    beta_val = pde_config.get("beta", [0.8, 0.3])
    
    nx_out = case_spec.get("nx", 50)
    ny_out = case_spec.get("ny", 50)
    
    # 2. Create mesh - use degree 2 elements for accuracy
    mesh_res = 64
    element_degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = cos(pi*x)*sin(pi*y)
    u_exact = ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term from manufactured solution
    # -eps * laplacian(u_exact) + beta . grad(u_exact) = f
    # laplacian of cos(pi*x)*sin(pi*y) = -pi^2*cos(pi*x)*sin(pi*y) - pi^2*cos(pi*x)*sin(pi*y) = -2*pi^2*cos(pi*x)*sin(pi*y)
    # So -eps * (-2*pi^2*cos(pi*x)*sin(pi*y)) = 2*eps*pi^2*cos(pi*x)*sin(pi*y)
    # grad(u_exact) = (-pi*sin(pi*x)*sin(pi*y), pi*cos(pi*x)*cos(pi*y))
    # beta . grad = beta[0]*(-pi*sin(pi*x)*sin(pi*y)) + beta[1]*(pi*cos(pi*x)*cos(pi*y))
    
    eps_c = fem.Constant(domain, default_scalar_type(eps_val))
    beta = fem.Constant(domain, default_scalar_type(np.array(beta_val)))
    
    # Source term derived from manufactured solution
    f_expr = (2.0 * eps_val * ufl.pi**2 * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta_val[0] * (-ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]))
              + beta_val[1] * (ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])))
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # Standard Galerkin
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_gal = f_expr * v * ufl.dx
    
    # SUPG terms
    # Residual applied to trial function: -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements, but for degree 2 it's nonzero
    # Strong residual of trial: -eps * div(grad(u)) + dot(beta, grad(u))
    # For SUPG test function modification: v_supg = tau * dot(beta, grad(v))
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    # If degree >= 2, include diffusion part of strong residual in SUPG
    if element_degree >= 2:
        a_supg += (-eps_c) * ufl.inner(ufl.div(ufl.grad(u)), v_supg) * ufl.dx
    
    a_form = a_gal + a_supg
    L_form = L_gal + L_supg
    
    # 5. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundaries
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_max_it": 2000,
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract on uniform grid
    xv = np.linspace(0, 1, nx_out)
    yv = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(xv, yv, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = xx.ravel()
    points[1, :] = yy.ravel()
    
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
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }