import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve_variant(N, degree, use_supg=True):
    comm = MPI.COMM_WORLD
    epsilon = 0.005
    beta_vec = [15.0, 7.0]
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    f = ufl.sin(10 * pi * x[0]) * ufl.sin(8 * pi * x[1])
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    if use_supg:
        h = ufl.CellDiameter(domain)
        beta_mag = ufl.sqrt(ufl.dot(beta, beta))
        Pe_local = beta_mag * h / (2.0 * epsilon)
        tau = h / (2.0 * beta_mag) * ufl.min_value(Pe_local / 3.0, 1.0)
        r_test = ufl.dot(beta, ufl.grad(v))
        
        if degree == 1:
            a += tau * ufl.dot(beta, ufl.grad(u)) * r_test * ufl.dx
            L += tau * f * r_test * ufl.dx
        else:
            # For higher order, include diffusion in residual
            a += tau * (-epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * r_test * ufl.dx
            L += tau * f * r_test * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-10",
            "ksp_max_it": "10000",
            "ksp_gmres_restart": "150",
        },
        petsc_options_prefix=f"test_{N}_{degree}_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx * ny, np.nan)
    pts_list = []
    cells_list = []
    idx_list = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points[:, i])
            cells_list.append(links[0])
            idx_list.append(i)
    
    if len(pts_list) > 0:
        vals = u_sol.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        u_values[idx_list] = vals.flatten()
    
    return u_values.reshape((nx, ny))

# Test different configurations
configs = [
    (64, 1, True, "P1+SUPG N=64"),
    (128, 1, True, "P1+SUPG N=128"),
    (256, 1, True, "P1+SUPG N=256"),
    (64, 2, True, "P2+SUPG N=64"),
    (128, 2, True, "P2+SUPG N=128"),
    (64, 2, False, "P2 no SUPG N=64"),
    (128, 2, False, "P2 no SUPG N=128"),
]

results = {}
for N, deg, supg, label in configs:
    t0 = time.time()
    u_grid = solve_variant(N, deg, supg)
    t1 = time.time()
    norm = np.sqrt(np.nanmean(u_grid**2))
    results[label] = u_grid
    print(f"{label}: norm={norm:.6e}, range=[{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}], time={t1-t0:.2f}s")

# Compare pairs
print("\n--- Differences between configurations ---")
ref = results["P1+SUPG N=256"]
for label, u_grid in results.items():
    diff = np.sqrt(np.nanmean((u_grid - ref)**2))
    print(f"  {label} vs P1+SUPG N=256: L2 diff = {diff:.6e}")
