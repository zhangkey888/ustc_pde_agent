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
    nu_val = 0.49
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E / (2.0 * (1.0 + nu_val))

    # Near-incompressible: need higher resolution and degree 2
    mesh_resolution = 64
    element_degree = 2

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", element_degree, (tdim,)))

    # Exact solution
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector([ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                              ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])])

    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(tdim)

    # Source term from manufactured solution: f = -div(sigma(u_exact))
    f = -ufl.div(sigma(u_exact))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions: u = u_exact on all boundaries
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
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

    u_magnitude = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2)
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]

    u_grid = u_magnitude.reshape((nx_eval, ny_eval))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "gamg",
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}