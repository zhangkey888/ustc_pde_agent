import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]
    mesh_res = 96
    elem_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    comm = MPI.COMM_WORLD
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    kappa = 1.0 + 0.3 * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(8*ufl.pi*x[1])
    f_expr = -ufl.div(kappa * ufl.grad(u_exact))
    g_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(g_func, boundary_dofs)
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}, petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    iterations = problem.solver.getIterationNumber()
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_values_clean = np.where(np.isnan(u_values), 0.0, u_values)
    u_values_all = np.zeros_like(u_values_clean)
    comm.Allreduce(u_values_clean, u_values_all, op=MPI.SUM)
    u_grid = u_values_all.reshape(ny_out, nx_out)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    error_L2_sq = fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx))
    error_L2 = np.sqrt(comm.allreduce(error_L2_sq, op=MPI.SUM))
    solver_info = {"mesh_resolution": mesh_res, "element_degree": elem_degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iterations}
    if comm.rank == 0:
        print(f"L2_ERROR: {error_L2:.6e}")
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    print(f"WALL_TIME: {elapsed:.4f}")
