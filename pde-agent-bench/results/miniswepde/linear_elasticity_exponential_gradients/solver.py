import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict = None) -> dict:
    t_start = time.time()

    # Material parameters
    E = 1.0
    nu_val = 0.33
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E / (2.0 * (1.0 + nu_val))

    N = 32
    degree = 3

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

    # Define exact solution using UFL for source term computation
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact_ufl = ufl.as_vector([
        ufl.exp(2 * x[0]) * ufl.sin(pi * x[1]),
        -ufl.exp(2 * x[1]) * ufl.sin(pi * x[0])
    ])

    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

    # Compute source term: f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact_ufl))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear form
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions: u = u_exact on all boundary
    u_bc = fem.Function(V)

    def u_exact_func(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = np.exp(2 * x[0]) * np.sin(np.pi * x[1])
        vals[1] = -np.exp(2 * x[1]) * np.sin(np.pi * x[0])
        return vals

    u_bc.interpolate(u_exact_func)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve with CG + AMG
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": "1e-12",
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()

    # Get solver iterations
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_grid = np.full((nx_eval * ny_eval,), np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # displacement magnitude = sqrt(ux^2 + uy^2)
        disp_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = disp_mag[idx]

    u_grid_2d = u_grid.reshape((nx_eval, ny_eval))

    result = {
        "u": u_grid_2d,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(iterations),
        }
    }

    return result
