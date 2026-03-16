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
    epsilon = 0.03
    beta = np.array([5.0, 2.0])
    
    # Mesh resolution and element degree
    N = 80
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian(u) = -pi^2 * sin(pi*x)*sin(2*pi*y) - 4*pi^2 * sin(pi*x)*sin(2*pi*y)
    #              = -5*pi^2 * sin(pi*x)*sin(2*pi*y)
    # So -eps * laplacian = eps * 5 * pi^2 * sin(pi*x)*sin(2*pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(2*pi*y), 2*pi*sin(pi*x)*cos(2*pi*y))
    # beta . grad(u) = 5*pi*cos(pi*x)*sin(2*pi*y) + 2*2*pi*sin(pi*x)*cos(2*pi*y)
    
    beta_ufl = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                               fem.Constant(domain, ScalarType(beta[1]))])
    eps_const = fem.Constant(domain, ScalarType(epsilon))

    f_expr = -eps_const * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_ufl, ufl.grad(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ h/(2|beta|) for large Pe
    # Use a simpler formula that works well:
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0), 1.0 - 1.0/Pe_cell, Pe_cell/3.0))

    # Residual applied to trial function (for linear SUPG, the strong-form operator on u)
    # Strong form residual: -eps*laplacian(u) + beta.grad(u) - f
    # For trial function, we use: beta.grad(u) (since laplacian of linear trial in SUPG test is often dropped for P1,
    # but for P2 we can include it)
    # Actually for SUPG with P2, including the diffusion term in the residual is beneficial
    # But ufl.div(ufl.grad(u)) for a TrialFunction requires careful handling
    # For the bilinear form, the SUPG adds: tau * (beta.grad(u)) * (beta.grad(v)) 
    # and for the RHS: tau * f * (beta.grad(v))
    # The diffusion part -eps*laplacian(u) in the residual tested with tau*beta.grad(v) 
    # For P2 elements, laplacian is piecewise linear (nonzero), but it's complex to include.
    # Standard approach: only include convection part in the operator for SUPG stabilization term.

    r_u = ufl.dot(beta_ufl, ufl.grad(u))  # convective part of residual for trial
    supg_test = tau * ufl.dot(beta_ufl, ufl.grad(v))

    a_supg = a + r_u * supg_test * ufl.dx
    L_supg = L + f_expr * supg_test * ufl.dx

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary
    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
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
        petsc_options_prefix="cdiff_"
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
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}