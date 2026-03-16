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
    beta = [3.0, 3.0]
    
    # Mesh resolution - need enough for high-frequency solution with Pe~85
    # The solution is sin(4*pi*x)*sin(3*pi*y), high frequency needs good resolution
    N = 80
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(4*pi*x)*sin(3*pi*y) = -(16*pi^2 + 9*pi^2)*sin(4*pi*x)*sin(3*pi*y) = -25*pi^2*u
    # So -eps * (-25*pi^2*u) = 25*eps*pi^2*u
    # beta . grad(u) = 3*4*pi*cos(4*pi*x)*sin(3*pi*y) + 3*3*pi*sin(4*pi*x)*cos(3*pi*y)
    #                = 12*pi*cos(4*pi*x)*sin(3*pi*y) + 9*pi*sin(4*pi*x)*cos(3*pi*y)
    
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    beta_ufl = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                               fem.Constant(domain, ScalarType(beta[1]))])
    eps_const = fem.Constant(domain, ScalarType(epsilon))

    f_ufl = -eps_const * laplacian_u_exact + ufl.dot(beta_ufl, grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp tau to be non-negative (for safety, though Pe is large here)
    # Actually, for Pe~85 per cell this is fine. Let's use a standard formula:
    # tau = h^2 / (4*eps) when diffusion dominated, tau = h/(2|beta|) when convection dominated
    # Optimal: tau = h/(2|beta|) * xi(Pe), xi(Pe) = coth(Pe) - 1/Pe
    # For large Pe, xi ≈ 1, so tau ≈ h/(2|beta|)
    tau_supg = h / (2.0 * beta_norm)

    # Residual of the strong form applied to trial function
    # For SUPG with linear problem, the residual operator on u is:
    # R(u) = -eps*laplacian(u) + beta.grad(u) - f
    # But for trial function (linear), we add:
    # a_supg = tau * (beta.grad(u)) * (beta.grad(v)) dx  (simplified SUPG for convection-dominated)
    # Full SUPG: tau * (-eps*lap(u) + beta.grad(u) - f) * (beta.grad(v)) dx
    # For P2 elements, lap(u) is nonzero, but for simplicity and since Pe is high,
    # the diffusion part of the residual is small. Let's include it properly for P2.
    
    # For the bilinear form, SUPG adds:
    # tau * (beta.grad(v)) * (-eps*div(grad(u)) + beta.grad(u)) dx
    # For the RHS:
    # tau * (beta.grad(v)) * f dx
    
    # Note: -eps*div(grad(u)) for P2 on triangles is piecewise constant (nonzero)
    
    a_supg = tau_supg * ufl.dot(beta_ufl, ufl.grad(v)) * (-eps_const * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))) * ufl.dx
    L_supg = tau_supg * ufl.dot(beta_ufl, ufl.grad(v)) * f_ufl * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(4 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
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
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }