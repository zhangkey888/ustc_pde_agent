import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    epsilon = pde["coefficients"]["diffusion"]
    beta_vec = pde["coefficients"]["velocity"]
    beta_np = np.array(beta_vec, dtype=np.float64)
    beta_norm = np.linalg.norm(beta_np)
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    N = 256
    domain = mesh.create_rectangle(comm, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [N, N], cell_type=mesh.CellType.quadrilateral)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    beta = ufl.as_vector(beta_np)
    f_expr = (epsilon * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[0] * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
              + beta[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_norm)
    a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
         + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx)
    L = f_expr * v * ufl.dx + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="convdiff_")
    u_sol = problem.solve()
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    if comm.size > 1:
        u_values_filled = np.where(np.isnan(u_values), 0.0, u_values)
        u_values_final = np.zeros_like(u_values)
        comm.Allreduce(u_values_filled, u_values_final, op=MPI.SUM)
        u_values = u_values_final
    u_grid = u_values.reshape(ny_out, nx_out)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(lambda xx: np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))
    error_L2 = fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx))
    error_L2 = np.sqrt(comm.allreduce(error_L2, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, Iterations: {iterations}")
    solver_info = {"mesh_resolution": N, "element_degree": degree, "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12, "iterations": iterations}
    return {"u": u_grid, "solver_info": solver_info}
