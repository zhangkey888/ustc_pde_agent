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
    
    epsilon = params.get("epsilon", 0.01)
    beta = params.get("beta", [-12.0, 6.0])
    
    # High Pe number => need SUPG stabilization and fine mesh
    N = 128
    degree = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(x)*sin(pi*y)
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Compute source term: f = -epsilon * laplacian(u) + beta . grad(u)
    # grad(u_exact) = (exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    # laplacian(u_exact) = exp(x)*sin(pi*y) - pi^2*exp(x)*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi^2)
    # So: -epsilon * laplacian = -epsilon * exp(x)*sin(pi*y)*(1 - pi^2)
    #     beta . grad = beta[0]*exp(x)*sin(pi*y) + beta[1]*exp(x)*pi*cos(pi*y)
    
    beta_vec = ufl.as_vector([default_scalar_type(beta[0]), default_scalar_type(beta[1])])
    
    f_expr = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_vec, ufl.grad(u_exact_ufl))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    
    # Stabilization parameter (classic formula)
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, 0.0))
    
    # SUPG residual: R(u) = -epsilon*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within each cell
    # So R(u) = beta.grad(u) - f
    R_u = ufl.dot(beta_vec, ufl.grad(u)) - f_expr
    
    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta_vec, ufl.grad(v))
    
    a += R_u * v_supg * ufl.dx
    # But R_u contains u (trial), so split:
    # R_u = beta.grad(u) - f
    # a_supg = tau * (beta.grad(u)) * (beta.grad(v)) * dx
    # L_supg = tau * f * (beta.grad(v)) * dx
    # We need to redo this properly since R_u has both u and f
    
    # Reset and redo properly
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG additions
    a += tau * ufl.dot(beta_vec, ufl.grad(u)) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    L += tau * f_expr * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    
    # Boundary conditions - apply exact solution on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
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
            "ksp_atol": "1e-14",
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
        },
        petsc_options_prefix="convdiff_"
    )
    uh = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Extract solution on 55x55 grid
    nx_out, ny_out = 55, 55
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
    
    for i in range(nx_out * ny_out):
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
            "stabilization": "SUPG",
        }
    }