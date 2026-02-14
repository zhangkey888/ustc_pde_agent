import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

comm = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType

def solve_with_f_expr(N, use_ufl=True):
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ('Lagrange', 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # UFL expression for source term
    f_ufl = ufl.sin(6*ufl.pi*x[0]) * ufl.sin(5*ufl.pi*x[1]) + \
            0.4 * ufl.sin(11*ufl.pi*x[0]) * ufl.sin(9*ufl.pi*x[1])
    
    if use_ufl:
        f = f_ufl
    else:
        # Interpolate
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: np.sin(6*np.pi*x[0]) * np.sin(5*np.pi*x[1]) + 
                                    0.4 * np.sin(11*np.pi*x[0]) * np.sin(9*np.pi*x[1]))
        f = f_func
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary condition
    tdim = domain.topology.dim
    fdim = tdim - 1
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0)
        ])
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    u_sol = fem.Function(V)
    problem = petsc.LinearProblem(a, L, bcs=[bc], u=u_sol,
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
        petsc_options_prefix='test_')
    u_sol = problem.solve()
    
    # Evaluate on grid
    nx, ny = 50, 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    points = np.zeros((3, nx * ny))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            points[0, idx] = x_vals[i]
            points[1, idx] = y_vals[j]
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
    u_values_all = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values_all[eval_map] = vals.flatten()
    u_grid = u_values_all.reshape((ny, nx))
    u_grid = np.nan_to_num(u_grid, nan=0.0)
    return u_grid

ref_data = np.load('oracle_output/reference.npz')
u_star = ref_data['u_star']

for use_ufl in [True, False]:
    print(f'Using UFL expression: {use_ufl}')
    for N in [32, 64, 128]:
        u = solve_with_f_expr(N, use_ufl)
        error = np.abs(u - u_star)
        max_error = error.max()
        print(f'  N={N}: max error = {max_error:.6e}')
