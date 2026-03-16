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
    epsilon = 0.2
    beta = [1.0, 0.5]
    mesh_resolution = 64
    element_degree = 2

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    # laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # So -eps * (-2*pi^2 * sin(pi*x)*sin(pi*y)) = 2*eps*pi^2 * sin(pi*x)*sin(pi*y)
    # grad(u) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    # beta . grad(u) = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    beta_vec = ufl.as_vector([beta[0], beta[1]])
    f_ufl = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_vec, ufl.grad(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form: epsilon * (grad u, grad v) + (beta . grad u, v)
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx

    # Linear form
    L = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ≈ for moderate Pe
    # Simple formula:
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                      1.0 - 1.0 / Pe_cell,
                                                      Pe_cell / 3.0))

    # SUPG residual: R(u) = -eps * laplacian(u) + beta . grad(u) - f
    # For trial function (linear), laplacian of linear Lagrange = 0 on each element
    # For degree 2, we can include it
    residual_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    residual_f = f_ufl

    # SUPG terms
    a_supg = tau * ufl.dot(beta_vec, ufl.grad(v)) * residual_u * ufl.dx
    L_supg = tau * ufl.dot(beta_vec, ufl.grad(v)) * residual_f * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # Boundary conditions - u = 0 on all boundaries (sin(pi*x)*sin(pi*y) = 0 on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
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
            "ksp_max_it": "1000",
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
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }