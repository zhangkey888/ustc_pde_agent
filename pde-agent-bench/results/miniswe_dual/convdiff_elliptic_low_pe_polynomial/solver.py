import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec=None):
    eps_val = 0.3
    beta_vec = [0.5, 0.3]
    nx_out, ny_out = 50, 50
    element_degree = 2
    N = 32
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = x[0]*(1-x[0])*x[1]*(1-x[1])
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = ufl.as_vector([PETSc.ScalarType(beta_vec[0]), PETSc.ScalarType(beta_vec[1])])
    f_expr = -eps_c*ufl.div(ufl.grad(u_exact_expr)) + ufl.dot(beta, ufl.grad(u_exact_expr))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = eps_c*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)*ufl.dx
    L = ufl.inner(f_expr, v)*ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: x[0]*(1-x[0])*x[1]*(1-x[1]))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol)}, petsc_options_prefix="cdiff_")
    u_sol = problem.solve()
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    return {"u": u_grid, "solver_info": {"mesh_resolution": N, "element_degree": element_degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": 0}}

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = XX*(1-XX)*YY*(1-YY)
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_error = np.nanmax(np.abs(u_grid - u_exact))
    print(f"Shape: {u_grid.shape}, Time: {elapsed:.3f}s, RMS: {error:.6e}, Max: {max_error:.6e}")
