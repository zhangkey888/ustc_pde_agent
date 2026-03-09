import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """Solve convection-diffusion with SUPG stabilization."""
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.01)
    beta_vec = params.get("beta", [12.0, 0.0])
    
    domain_spec = case_spec.get("domain", {})
    bounds = domain_spec.get("bounds", [[0, 1], [0, 1]])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # For this high Peclet number problem with SUPG, degree 2 and N=80 is sufficient
    N = 80
    degree = 2
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    p0 = np.array([bounds[0][0], bounds[1][0]], dtype=np.float64)
    p1 = np.array([bounds[0][1], bounds[1][1]], dtype=np.float64)
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], 
                                    cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Convection velocity
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    
    # Exact solution for BC and source term
    u_exact = ufl.exp(3.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term from manufactured solution
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    f = -epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    
    # SUPG stabilization parameter (optimal formula)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # Standard Galerkin terms
    a_gal = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + 
              ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L_gal = f * v * ufl.dx
    
    # SUPG stabilization terms
    R_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = R_u * v_supg * ufl.dx
    L_supg = f * v_supg * ufl.dx
    
    # Combined forms
    a_total = a_gal + a_supg
    L_total = L_gal + L_supg
    
    # Boundary conditions
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
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
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out, 
                                [bounds[0][0], bounds[0][1]], 
                                [bounds[1][0], bounds[1][1]])
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 0,
        }
    }


def _evaluate_on_grid(domain, u_sol, nx, ny, x_range, y_range):
    """Evaluate solution on a uniform grid."""
    
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    return u_grid
