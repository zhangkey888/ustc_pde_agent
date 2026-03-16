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
    
    # Mesh resolution - need fine enough for accuracy but fast enough
    N = 128
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Coordinates
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    # f = -eps * laplacian(u_exact) + beta . grad(u_exact)
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    f_ufl = -eps_c * laplacian_u_exact + ufl.dot(beta, grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    
    # SUPG stabilization parameter
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # Classical formula with coth
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe)
    # For high Pe, tau ~ h / (2*|beta|)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # Residual applied to trial function (for linear SUPG)
    # Strong form residual operator on u: -eps*laplacian(u) + beta.grad(u) - f
    # For linear problem with trial function, we use:
    # SUPG adds: tau * (beta.grad(u) - f) * (beta.grad(v)) dx
    # Note: we skip the diffusion part of the residual in the test function modification
    # since for linear elements the Laplacian of u_h is zero element-wise,
    # but for degree 2 it's not. However, standard SUPG typically only uses
    # the advection part in the stabilization for simplicity.
    
    # SUPG stabilization terms
    r_lhs = ufl.dot(beta, ufl.grad(u))  # advection part of residual (LHS)
    r_rhs = f_ufl  # source term
    
    # For degree 2, we can include the diffusion term too
    # But the standard approach: stabilization test function = tau * beta.grad(v)
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    
    a_supg = a_std + r_lhs * supg_test * ufl.dx
    L_supg = L_std + r_rhs * supg_test * ufl.dx

    # Crosswind diffusion (additional stabilization for high Pe)
    # Add isotropic artificial diffusion in crosswind direction
    # This helps with oscillations perpendicular to the flow
    u_h_temp = fem.Function(V)  # We'll do iterative approach if needed
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary
    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a_supg, L_supg, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="solve_"
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
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }