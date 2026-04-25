import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    kappa = 1.0
    mesh_res = 128
    element_degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Exact solution: u = sin(pi*x)*sin(pi*y) + 0.2*sin(5*pi*x)*sin(4*pi*y)
    # Source: f = 2*pi^2*sin(pi*x)*sin(pi*y) + 8.2*pi^2*sin(5*pi*x)*sin(4*pi*y)
    def u_exact_lambda(x):
        return (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) 
                + 0.2 * np.sin(5 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))

    def f_lambda(x):
        return (2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) 
                + 8.2 * np.pi**2 * np.sin(5 * np.pi * x[0]) * np.sin(4 * np.pi * x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_func = fem.Function(V)
    f_func.interpolate(f_lambda)

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx

    # Dirichlet BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_lambda)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Direct solver
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = problem.solver.getIterationNumber()

    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": iterations,
        }
    }
