import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    eps_val = case_spec["pde"]["coefficients"]["epsilon"]
    beta_vec = case_spec["pde"]["coefficients"]["beta"]
    
    # Mesh resolution - need fine enough for accuracy but fast enough
    N = 80
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Coordinates
    x = ufl.SpatialCoordinate(domain)

    # Exact solution for BCs and source term
    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # u = exp(x)*sin(pi*y)
    # grad(u) = (exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    # laplacian(u) = exp(x)*sin(pi*y) - exp(x)*pi^2*sin(pi*y) = exp(x)*sin(pi*y)*(1 - pi^2)
    # f = -eps * exp(x)*sin(pi*y)*(1-pi^2) + beta[0]*exp(x)*sin(pi*y) + beta[1]*exp(x)*pi*cos(pi*y)

    eps_c = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))

    f_ufl = (-eps_val * (1.0 - ufl.pi**2) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
             + beta_vec[0] * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
             + beta_vec[1] * ufl.exp(x[0]) * ufl.pi * ufl.cos(ufl.pi * x[1]))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp tau to be non-negative via a simpler approach:
    # For high Pe, tau ≈ h/(2*|beta|)
    # Use the standard formula:
    tau_supg = h * h / (4.0 * eps_c + 2.0 * h * beta_norm)

    # Residual applied to trial function (for linear SUPG)
    # R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0 on each element
    # For quadratic elements, we still include it
    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_ufl

    # SUPG test function modification
    v_supg = ufl.dot(beta, ufl.grad(v))

    a_supg = a + tau_supg * ufl.inner(r_u, v_supg) * ufl.dx
    # Wait, r_u contains f_ufl which goes to RHS. Let me split properly.
    # SUPG adds: tau * (L_adv(u) - f) * (beta . grad(v)) dx
    # where L_adv(u) = -eps*lap(u) + beta.grad(u)
    # So bilinear part: tau * (-eps*lap(u) + beta.grad(u)) * (beta.grad(v)) dx
    # Linear part: tau * f * (beta.grad(v)) dx

    a_stab = tau_supg * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * v_supg * ufl.dx
    L_stab = tau_supg * f_ufl * v_supg * ufl.dx

    a_total = a + a_stab
    L_total = L + L_stab

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
    u_bc.interpolate(lambda x: np.exp(x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "2000",
            "ksp_gmres_restart": "100",
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

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}