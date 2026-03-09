import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec=None):
    if case_spec is None:
        case_spec = {}
    k_val = 2.0
    nx_out = 50
    ny_out = 50
    pde_spec = case_spec.get('pde', {})
    params = pde_spec.get('parameters', {})
    if 'k' in params:
        k_val = float(params['k'])
    output_spec = case_spec.get('output', {})
    if 'nx' in output_spec:
        nx_out = int(output_spec['nx'])
    if 'ny' in output_spec:
        ny_out = int(output_spec['ny'])
    comm = MPI.COMM_WORLD
    N = 32
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    k2 = k_val * k_val
    f_expr = (2.0*x[1]*(1.0-x[1]) + 2.0*x[0]*(1.0-x[0]) - k2*x[0]*(1.0-x[0])*x[1]*(1.0-x[1]))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - k2*ufl.inner(u, v)*ufl.dx
    L = f_expr*v*ufl.dx
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": str(rtol), "ksp_max_it": "1000"},
        petsc_options_prefix="helmholtz_")
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((3, nx_out*ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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
    u_values = np.full(nx_out*ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    return {"u": u_grid, "solver_info": {"mesh_resolution": N, "element_degree": degree, "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "iterations": iterations}}

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    nx, ny = u_grid.shape
    x_c = np.linspace(0, 1, nx)
    y_c = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_c, y_c, indexing='ij')
    u_exact = X*(1-X)*Y*(1-Y)
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    max_err = np.nanmax(np.abs(u_grid - u_exact))
    print(f"Shape: {u_grid.shape}, Time: {elapsed:.3f}s")
    print(f"RMS error: {error:.6e}, Max error: {max_err:.6e}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
    print(f"Info: {result['solver_info']}")
