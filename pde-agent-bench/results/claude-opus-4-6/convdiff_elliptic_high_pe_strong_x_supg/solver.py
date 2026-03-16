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
    epsilon = case_spec["pde"]["params"]["epsilon"]
    beta_vec = case_spec["pde"]["params"]["beta"]
    
    # Mesh resolution and element degree
    N = 64
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    f_ufl = -eps_c * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Optimal SUPG parameter
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # Residual applied to trial function (for SUPG)
    # R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian of trial function is zero within elements
    # For higher order, we still use the strong-form residual projected
    # SUPG test function modification
    v_supg = v + tau * ufl.dot(beta, ufl.grad(v))

    # Standard Galerkin
    a_gal = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = f_ufl * v * ufl.dx

    # SUPG additional terms
    # The SUPG stabilization adds: tau * (beta . grad(v)) * (residual)
    # residual of u: -eps * laplacian(u) + beta . grad(u) - f
    # For the bilinear form: tau * (beta . grad(v)) * (-eps * laplacian(u) + beta . grad(u))
    # For degree >= 2, laplacian is nonzero but for simplicity and robustness,
    # we use the streamline derivative form:
    a_supg = tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L_supg = tau * f_ufl * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a = a_gal + a_supg
    L = L_gal + L_supg

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary: u = sin(pi*x)*sin(pi*y) = 0 on boundary of [0,1]^2
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "2000",
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
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

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
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": iterations,
        }
    }