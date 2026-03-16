import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse parameters
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [10.0, 5.0])
    
    domain_spec = case_spec.get("domain", {})
    extents = domain_spec.get("extents", [[0, 1], [0, 1]])
    
    # Mesh resolution - use fine mesh for accuracy
    N = 128
    element_degree = 1
    
    # Create mesh
    p0 = np.array([extents[0][0], extents[1][0]])
    p1 = np.array([extents[0][1], extents[1][1]])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Parameters
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    # Source term: Gaussian
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Boundary conditions: u = 0 on all boundaries (default for no exact solution case)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    
    # SUPG stabilization parameter (optimal for 1D)
    tau = h / (2.0 * beta_norm) * (ufl.cosh(Pe_cell) / ufl.sinh(Pe_cell) - 1.0 / Pe_cell)
    
    # Standard Galerkin terms
    a_standard = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_standard = f_expr * v * ufl.dx
    
    # SUPG stabilization terms
    # Residual applied to test function: tau * (beta · ∇v) * R
    # For linear problem: R = -eps*Δu + beta·∇u - f
    # Since we use linear elements, Δu = 0 within elements
    # So the SUPG modification is:
    # a_supg = tau * (beta·∇v) * (beta·∇u) dx
    # L_supg = tau * (beta·∇v) * f dx
    
    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * ufl.dot(beta, ufl.grad(u)) * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx
    
    a = a_standard + a_supg
    L = L_standard + L_supg
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(extents[0][0], extents[0][1], nx_out)
    ys = np.linspace(extents[1][0], extents[1][1], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
        vals = u_sol.eval(pts_arr, cells_arr)
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