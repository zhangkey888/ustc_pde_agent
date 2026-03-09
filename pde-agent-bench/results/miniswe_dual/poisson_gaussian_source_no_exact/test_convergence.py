import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve_at_resolution(N, degree):
    comm = MPI.COMM_WORLD
    nx_output = 50
    ny_output = 50
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.25)**2 + (x[1] - 0.75)**2))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
        petsc_options_prefix="test_"
    )
    u_sol = problem.solve()
    
    # Evaluate on grid
    x_coords = np.linspace(0, 1, nx_output)
    y_coords = np.linspace(0, 1, ny_output)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_3d = np.zeros((nx_output * ny_output, 3))
    points_3d[:, 0] = xx.flatten()
    points_3d[:, 1] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_output * ny_output, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx_output, ny_output))

# Test convergence
configs = [(32, 2), (64, 2), (80, 2), (128, 2), (64, 3), (128, 3)]
results = {}
for N, deg in configs:
    t0 = time.time()
    u_grid = solve_at_resolution(N, deg)
    elapsed = time.time() - t0
    results[(N, deg)] = u_grid
    print(f"N={N:3d}, deg={deg}: max={np.nanmax(u_grid):.8e}, time={elapsed:.3f}s")

# Compare solutions
ref = results[(128, 3)]
for (N, deg), u_grid in results.items():
    diff = np.nanmax(np.abs(u_grid - ref))
    l2_diff = np.sqrt(np.nanmean((u_grid - ref)**2))
    print(f"N={N:3d}, deg={deg}: max_diff_vs_ref={diff:.6e}, l2_diff={l2_diff:.6e}")
