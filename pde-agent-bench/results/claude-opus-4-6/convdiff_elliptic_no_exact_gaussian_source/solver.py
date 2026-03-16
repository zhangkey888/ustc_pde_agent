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
    beta = [2.0, 1.0]
    mesh_resolution = 128
    element_degree = 1

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Source term
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))

    # Velocity field
    beta_vec = ufl.as_vector([ScalarType(beta[0]), ScalarType(beta[1])])

    # Diffusion coefficient
    eps_const = fem.Constant(domain, ScalarType(epsilon))

    # Standard Galerkin bilinear form
    a_standard = eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
                 + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    L_standard = ufl.inner(f_expr, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_const)
    # Stabilization parameter
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                      1.0 - 1.0 / Pe_cell,
                                                      0.0))

    # Residual applied to trial function (for linear SUPG, the strong-form operator on u)
    # Strong form residual: -eps * laplacian(u) + beta . grad(u) - f
    # For linear elements, laplacian(u) = 0, so residual simplifies
    R_u = ufl.dot(beta_vec, ufl.grad(u)) - f_expr  # -eps*lap(u) = 0 for P1

    # SUPG terms
    a_supg = tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)),
                              ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx
    L_supg = tau * ufl.inner(f_expr, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx

    a = a_standard + a_supg
    L = L_standard + L_supg

    # Boundary conditions: u = 0 on all boundaries
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

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
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }