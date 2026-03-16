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
    
    # Mesh resolution - need enough resolution for accuracy but within time limit
    # Pe ~ 84.9, so we need SUPG stabilization + reasonable mesh
    N = 80
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Compute source term from manufactured solution
    # -eps * laplacian(u) + beta . grad(u) = f
    beta_vec = ufl.as_vector([ScalarType(beta[0]), ScalarType(beta[1])])
    
    f_ufl = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_vec, ufl.grad(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Standard Galerkin
    a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta_vec, ufl.grad(u)) * v) * ufl.dx
    L = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # Optimal tau
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # SUPG residual: strong form residual tested with beta . grad(v)
    # Strong form operator applied to trial: -eps * laplacian(u) + beta . grad(u)
    # For linear elements, laplacian of u is zero within elements, but for degree 2 it's not
    # We use the full residual for SUPG
    r_strong = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))
    supg_test = ufl.dot(beta_vec, ufl.grad(v))

    a_supg = a + tau * r_strong * supg_test * ufl.dx
    L_supg = L + tau * f_ufl * supg_test * ufl.dx

    # Boundary conditions - u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)

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

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}