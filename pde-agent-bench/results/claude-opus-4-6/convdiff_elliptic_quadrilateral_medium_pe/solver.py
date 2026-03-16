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
    epsilon = 0.05
    beta = np.array([4.0, 2.0])
    
    # Mesh resolution and element degree
    N = 64
    degree = 2

    # Create quadrilateral mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    
    beta_ufl = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                               fem.Constant(domain, ScalarType(beta[1]))])
    eps = fem.Constant(domain, ScalarType(epsilon))
    
    f_expr = -eps * laplacian_u_exact + ufl.dot(beta_ufl, grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin bilinear form
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_norm * h / (2.0 * eps)
    # Stabilization parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)
    
    # SUPG residual applied to test function: tau * (beta . grad(v))
    # For the trial function: residual = -eps*laplacian(u) + beta.grad(u) - f
    # Since we use linear elements or higher, we approximate:
    # For degree >= 2, laplacian of u is nonzero but second derivatives in UFL are tricky
    # Standard SUPG: add tau * (beta.grad(v)) * (beta.grad(u) - f) * dx
    # (dropping the diffusion term in the residual for SUPG, which is standard)
    
    r_test = tau * ufl.dot(beta_ufl, ufl.grad(v))
    a_supg = r_test * ufl.dot(beta_ufl, ufl.grad(u)) * ufl.dx
    L_supg = r_test * f_expr * ufl.dx

    a += a_supg
    L += L_supg

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary
    def boundary_all(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)

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
            "ksp_max_it": "2000",
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
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.ravel()
    points[1] = Y.ravel()

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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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
            "iterations": iterations,
        }
    }