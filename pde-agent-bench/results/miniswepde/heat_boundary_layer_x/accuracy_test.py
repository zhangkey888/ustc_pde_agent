import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def u_exact(x, t):
    return np.exp(-t) * np.exp(5*x[0]) * np.sin(np.pi*x[1])

def f_source(x, t):
    u = np.exp(-t) * np.exp(5*x[0]) * np.sin(np.pi*x[1])
    du_dt = -u
    laplacian_u = (25 - np.pi**2) * u
    return du_dt - laplacian_u

def solve_config(N, degree, dt_value, t_end=0.08):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    def boundary_marker(x):
        return np.logical_or.reduce([
            np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
        ])
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    # Time stepping
    u_n = fem.Function(V)
    u = fem.Function(V)
    u_n.interpolate(lambda x: u_exact(x, 0.0))
    u.x.array[:] = u_n.x.array
    
    v = ufl.TestFunction(V)
    dt = fem.Constant(domain, ScalarType(dt_value))
    kappa = fem.Constant(domain, ScalarType(1.0))
    
    n_steps = int(np.ceil(t_end / dt_value))
    actual_dt = t_end / n_steps
    dt.value = actual_dt
    
    for step in range(n_steps):
        t = (step + 1) * actual_dt
        u_bc.interpolate(lambda x: u_exact(x, t))
        bc = fem.dirichletbc(u_bc, dofs)
        
        t_mid = t - actual_dt/2
        f_func = fem.Function(V)
        f_func.interpolate(lambda x: f_source(x, t_mid))
        
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(u_trial, v) * ufl.dx + dt * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="test_"
        )
        u_sol = problem.solve()
        
        u.x.array[:] = u_sol.x.array
        u_n.x.array[:] = u.x.array
    
    # Compute error on 50x50 grid
    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0
    
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
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    u_exact_grid = u_exact(np.array([X, Y]), t_end)
    
    error_mask = ~np.isnan(u_grid)
    rms_error = np.sqrt(np.mean((u_grid[error_mask] - u_exact_grid[error_mask])**2))
    
    return rms_error, actual_dt, n_steps

# Test configurations
configs = [
    (128, 1, 0.002),
    (192, 1, 0.001),
    (256, 1, 0.001),
    (128, 2, 0.001),
]

print("Testing configurations...")
for N, degree, dt in configs:
    start = time.time()
    error, actual_dt, steps = solve_config(N, degree, dt)
    elapsed = time.time() - start
    print(f"N={N}, degree={degree}, dt={actual_dt:.4f}, steps={steps}: error={error:.2e}, time={elapsed:.1f}s")
