import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Material parameters
    E = 1.0
    nu_val = 0.3
    mu = E / (2.0 * (1.0 + nu_val))
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

    # Mesh resolution - use quadrilateral mesh as specified
    N = 64
    element_degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    gdim = domain.geometry.dim

    # Vector function space
    V = fem.functionspace(domain, ("Lagrange", element_degree, (gdim,)))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Exact solution
    u_exact_expr = ufl.as_vector([
        ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1]),
        ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])

    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

    # Compute source term from manufactured solution
    f_expr = -ufl.div(sigma(u_exact_expr))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions - all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": "1e-10",
            "ksp_atol": "1e-12",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

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

    u_values = np.full((points.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute displacement magnitude
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx_eval, ny_eval))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "gamg",
        "rtol": 1e-10,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}