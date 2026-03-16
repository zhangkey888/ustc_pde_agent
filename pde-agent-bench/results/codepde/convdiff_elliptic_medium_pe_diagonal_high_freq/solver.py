import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.05)
    beta = params.get("beta", [3.0, 3.0])
    
    # Grid for output
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Mesh resolution - use higher resolution for high Peclet number
    N = 128
    element_degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # 4. Manufactured solution and source term
    u_exact_ufl = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    # f = -epsilon * laplacian(u) + beta . grad(u)
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f_expr = -epsilon * laplacian_u_exact + ufl.dot(beta_vec, grad_u_exact)
    
    # 5. Variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    h = ufl.CellDiameter(domain)
    
    # Peclet number based element size
    beta_mag = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_h = beta_mag * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter
    # tau = h / (2 * |beta|) * (coth(Pe_h) - 1/Pe_h)
    # For high Pe, coth(Pe) - 1/Pe ≈ 1, so tau ≈ h/(2|beta|)
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_mag) * (1.0 - 1.0 / Pe_h)
    # Clamp tau to be non-negative via conditional
    tau = ufl.conditional(ufl.gt(Pe_h, 1.0), h / (2.0 * beta_mag), h * h / (4.0 * epsilon))
    
    # Standard Galerkin terms
    a_standard = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
                 + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization: add (R(u), tau * beta . grad(v))
    # R(u) = -epsilon * laplacian(u) + beta . grad(u) - f
    # For trial function (linear): R(u) = -epsilon * div(grad(u)) + beta . grad(u) - f
    # But laplacian of trial function is tricky with linear elements
    # SUPG residual for the test function modification:
    # test_supg = v + tau * beta . grad(v)
    
    r_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a_supg = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
             + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx \
             + ufl.dot(beta_vec, ufl.grad(u)) * r_supg * ufl.dx
    
    # For P2 elements, we can include the diffusion part of SUPG
    # -epsilon * div(grad(u)) is nonzero for P2
    a_supg += -epsilon * ufl.div(ufl.grad(u)) * r_supg * ufl.dx
    
    L_supg = f_expr * v * ufl.dx + f_expr * r_supg * ufl.dx
    
    a = a_supg
    L = L_supg
    
    # 6. Boundary conditions
    # u = g = sin(4*pi*x)*sin(3*pi*y) on boundary
    # On the unit square boundary, sin(4*pi*x)*sin(3*pi*y) = 0
    # because at x=0,1: sin(4*pi*0)=0, sin(4*pi*1)=0
    # and at y=0,1: sin(3*pi*0)=0, sin(3*pi*1)=0
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))
    
    bc = fem.dirichletbc(u_bc, dofs)
    
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
    iterations = problem.solver.getIterationNumber()
    
    # 8. Extract solution on uniform grid
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }