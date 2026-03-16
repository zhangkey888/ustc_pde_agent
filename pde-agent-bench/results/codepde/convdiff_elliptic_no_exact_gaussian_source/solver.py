import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    
    epsilon = pde_config.get("epsilon", 0.05)
    beta = pde_config.get("beta", [2.0, 1.0])
    
    # Mesh resolution - use fine mesh for accuracy
    N = 128
    element_degree = 1
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: Gaussian
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))
    
    # Velocity vector
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    # Standard Galerkin terms
    a_standard = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    )
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization
    # Element size
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    
    # Stabilization parameter (standard formula)
    Pe_local = beta_norm * h / (2.0 * epsilon)
    # Optimal tau with coth formula approximation
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) - 1/Pe ≈ 1, so tau ≈ h/(2|beta|)
    # Use a smooth approximation
    tau = h / (2.0 * beta_norm + 1e-10) * (1.0 - 1.0 / (Pe_local + 1e-10))
    # Clamp tau to be non-negative using a simpler expression
    # Alternative: use the standard definition
    tau = h * h / (4.0 * epsilon + 2.0 * beta_norm * h)
    
    # SUPG test function modification: v_supg = beta . grad(v)
    v_supg = ufl.dot(beta_vec, ufl.grad(v))
    
    # Residual applied to trial function (for linear PDE, residual of u is the PDE operator)
    # -epsilon * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 within elements, so residual simplifies
    R_u = ufl.dot(beta_vec, ufl.grad(u)) - f_expr  # laplacian term vanishes for P1
    
    a_supg = tau * ufl.dot(beta_vec, ufl.grad(u)) * v_supg * ufl.dx
    L_supg = tau * f_expr * v_supg * ufl.dx
    
    a_total = a_standard + a_supg
    L_total = L_standard + L_supg
    
    # 5. Boundary conditions (u = 0 on all boundaries)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 6. Solve
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
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
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
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }