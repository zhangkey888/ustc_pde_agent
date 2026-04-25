import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 64
    degree = 3
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    # f = -div(grad(u)) = -(u_xx + u_yy)
    # u_xx = -pi^2 * exp(6y) sin(pi x)
    # u_yy = 36 * exp(6y) sin(pi x)
    # -lap = (pi^2 - 36) * exp(6y) sin(pi x)
    f_expr = (ufl.pi**2 - 36.0) * ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Dirichlet BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    petsc_opts = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_opts,
        petsc_options_prefix="poisson_bl_",
    )
    u_sol = problem.solve()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    cells = np.empty(pts.shape[0], dtype=np.int32)
    for i in range(pts.shape[0]):
        links = coll.links(i)
        cells[i] = links[0] if len(links) > 0 else 0

    vals = u_sol.eval(pts, cells).reshape(ny_out, nx_out)

    return {
        "u": vals,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        },
    }
