import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    params = pde_config.get("params", {})
    
    epsilon = params.get("epsilon", 0.2)
    beta = params.get("beta", [1.0, 0.5])
    
    # Grid for output
    nx_out = 50
    ny_out = 50
    
    # Mesh resolution and element degree
    mesh_resolution = 80
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Velocity vector
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Source term: f = -epsilon * laplacian(u_exact) + beta . grad(u_exact)
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # So -epsilon * laplacian = epsilon * 2*pi^2*sin(pi*x)*sin(pi*y)
    # grad(u_exact) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    f_expr = (epsilon * 2.0 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + beta[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
              + beta[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    h = ufl.CellDiameter(domain)
    
    # Péclet number based element size
    beta_mag = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_h = beta_mag * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter
    tau = h / (2.0 * beta_mag) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)
    
    # Standard Galerkin terms
    a_standard = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization terms
    # Residual operator applied to trial function: -epsilon*laplacian(u) + beta.grad(u)
    # For linear elements, laplacian is zero within elements, but for degree 2 it's not
    R_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    
    # SUPG test function modification
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = ufl.inner(R_u, v_supg) * ufl.dx
    L_supg = f_expr * v_supg * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # 6. Boundary conditions (u = sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 7. Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
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
    
    # 8. Extract solution on uniform grid
    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')
    
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
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }