import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    comm = MPI.COMM_WORLD

    # Extract case parameters
    kappa = float(case_spec["pde"].get("kappa", 5.0))
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Balanced settings for speed and accuracy
    mesh_res = 120
    element_degree = 3

    # Create mesh on unit square
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    # Exact solution: u = cos(2*pi*x)*cos(3*pi*y)
    u_exact = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])

    # Source term f = kappa * 13 * pi^2 * cos(2*pi*x)*cos(3*pi*y)
    f_expr = kappa * 13 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])

    # Bilinear form
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # Linear form
    L = f_expr * v * ufl.dx

    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V, dtype=ScalarType)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points, dtype=ScalarType)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="poisson_"
    )

    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = problem.solver.getIterationNumber()

    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

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

    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
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
