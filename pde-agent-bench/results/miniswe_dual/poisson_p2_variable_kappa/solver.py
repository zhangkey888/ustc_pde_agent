import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    N = 48
    element_degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    kappa = 1.0 + 0.4 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    u_exact_ufl = ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1])
    f = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol), "ksp_max_it": "2000"}, petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.zeros((nx_out*ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    u_values = np.full(points_3d.shape[0], np.nan)
    pts_proc = []
    cells_proc = []
    emap = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_proc.append(points_3d[i])
            cells_proc.append(links[0])
            emap.append(i)
    if len(pts_proc) > 0:
        vals = u_sol.eval(np.array(pts_proc), np.array(cells_proc, dtype=np.int32))
        u_values[emap] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    return {"u": u_grid, "solver_info": {"mesh_resolution": N, "element_degree": element_degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iterations}}
