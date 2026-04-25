import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    mesh_res = 256
    elem_deg = 2
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    x = ufl.SpatialCoordinate(domain)
    kappa_expr = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol, "ksp_atol": 1e-12, "ksp_max_it": 1000},
        petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    N = nx_out * ny_out
    pts = np.zeros((N, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    cells = np.full(N, -1, dtype=np.int32)
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells[i] = links[0]
            mask[i] = True
    u_grid = np.zeros((ny_out, nx_out))
    valid_pts = pts[mask]
    valid_cells = cells[mask]
    if len(valid_pts) > 0:
        vals = u_sol.eval(valid_pts, valid_cells)
        u_flat = np.zeros(N)
        u_flat[mask] = vals.flatten()
        u_grid = u_flat.reshape(ny_out, nx_out)
    solver_info = {"mesh_resolution": mesh_res, "element_degree": elem_deg, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iterations}
    return {"u": u_grid, "solver_info": solver_info}
