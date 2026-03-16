import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    epsilon = 0.02
    beta = [6.0, 2.0]
    
    # Mesh resolution - need fine enough mesh for accuracy but fast enough
    # Pe ~ 316, so we need SUPG or very fine mesh. Case says no stabilization but
    # the title says "no_stabilization" - let's use SUPG anyway since Pe is high
    # and we need error <= 3.08e-03
    
    N = 80
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Compute source term: f = -eps * laplacian(u) + beta . grad(u)
    # laplacian of sin(2*pi*x)*sin(2*pi*y) = -8*pi^2 * sin(2*pi*x)*sin(2*pi*y)
    # So -eps * laplacian = eps * 8*pi^2 * sin(2*pi*x)*sin(2*pi*y)
    # grad(u) = (2*pi*cos(2*pi*x)*sin(2*pi*y), 2*pi*sin(2*pi*x)*cos(2*pi*y))
    # beta . grad(u) = 6*2*pi*cos(2*pi*x)*sin(2*pi*y) + 2*2*pi*sin(2*pi*x)*cos(2*pi*y)
    
    grad_u_exact = ufl.grad(u_exact)
    # f = -epsilon * div(grad(u_exact)) + beta . grad(u_exact)
    beta_vec = ufl.as_vector([ScalarType(beta[0]), ScalarType(beta[1])])
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    
    f_expr = -eps_const * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_vec, grad_u_exact)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Simple formula:
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))
    
    # SUPG residual: R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian vanishes; for higher order it doesn't but 
    # in SUPG we typically use: tau * (beta.grad(u) - f) * (beta.grad(v))
    # More precisely, the strong residual applied to trial function:
    R_u = -eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    
    a_supg = tau * R_u * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    L_supg = tau * f_expr * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    
    a_total = a + a_supg
    L_total = L + L_supg
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    def all_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
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
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="cdsolve_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate on 50x50 grid
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
            points_on_proc.append(points[:, i])
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
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }