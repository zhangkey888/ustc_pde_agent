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
    E = 1.0
    nu_val = 0.3
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    mu = E / (2.0 * (1.0 + nu_val))

    # Mesh
    nx = ny = 64
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Vector function space
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))

    # Boundary conditions
    # We need to figure out what "boundary driven" means with no exact solution.
    # The case is "no_exact_boundary_driven" with f=0. 
    # A common setup: apply non-trivial Dirichlet BCs on parts of the boundary.
    # Typical: fix bottom, apply displacement on top.

    # Bottom boundary: u = (0, 0) (fixed)
    def bottom(x):
        return np.isclose(x[1], 0.0)

    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    u_zero = fem.Function(V)
    u_zero.interpolate(lambda x: np.zeros((2, x.shape[1])))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs)

    # Top boundary: u = (0.1, 0.0) (shear)
    def top(x):
        return np.isclose(x[1], 1.0)

    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.stack([np.full_like(x[0], 0.1), np.zeros_like(x[0])]))
    bc_top = fem.dirichletbc(u_top, top_dofs)

    # Left boundary: u = (0, 0)
    def left(x):
        return np.isclose(x[0], 0.0)

    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.zeros((2, x.shape[1])))
    bc_left = fem.dirichletbc(u_left, left_dofs)

    # Right boundary: u = (0.1, 0.0) linearly varying with y? 
    # Actually let's keep it simple: apply BCs on all boundaries.
    # Bottom fixed, top sheared, left and right: linear interpolation
    def right(x):
        return np.isclose(x[0], 1.0)

    right_facets = mesh.locate_entities_boundary(domain, fdim, right)
    right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.stack([0.1 * x[1], np.zeros_like(x[0])]))
    bc_right = fem.dirichletbc(u_right, right_dofs)

    # Update left BC to also be linear
    u_left.interpolate(lambda x: np.stack([0.1 * x[1], np.zeros_like(x[0])]))

    bcs = [bc_bottom, bc_top, bc_left, bc_right]

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(tdim)

    f = fem.Constant(domain, ScalarType((0.0, 0.0)))

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Solve
    ksp_type = "cg"
    pc_type = "gamg"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
            "ksp_monitor": None,
        },
        petsc_options_prefix="elasticity_",
    )
    u_sol = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Evaluate on 50x50 grid
    n_eval = 50
    xs = np.linspace(0.0, 1.0, n_eval)
    ys = np.linspace(0.0, 1.0, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, n_eval * n_eval))
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

    u_magnitude = np.full(n_eval * n_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D vector
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]

    u_grid = u_magnitude.reshape((n_eval, n_eval))

    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }