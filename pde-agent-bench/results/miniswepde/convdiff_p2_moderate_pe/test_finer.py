import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
import time

comm = MPI.COMM_WORLD
eps = 0.03
beta = np.array([5.0, 2.0])

def exact_solution(x):
    return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])

def source_term(x):
    x_coord = x[0]
    y_coord = x[1]
    sin_pi_x = np.sin(np.pi * x_coord)
    sin_2pi_y = np.sin(2 * np.pi * y_coord)
    cos_pi_x = np.cos(np.pi * x_coord)
    cos_2pi_y = np.cos(2 * np.pi * y_coord)
    laplacian_u = -5 * (np.pi**2) * sin_pi_x * sin_2pi_y
    grad_u_x = np.pi * cos_pi_x * sin_2pi_y
    grad_u_y = 2 * np.pi * sin_pi_x * cos_2pi_y
    return -eps * laplacian_u + beta[0] * grad_u_x + beta[1] * grad_u_y

for N in [128, 256]:
    for degree in [2, 3]:
        start = time.time()
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # BC
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Forms
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        beta_ufl = ufl.as_vector([beta[0], beta[1]])
        a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) 
             + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v)) * ufl.dx
        f = fem.Function(V)
        f.interpolate(lambda x: source_term(x))
        L = ufl.inner(f, v) * ufl.dx
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
            petsc_options_prefix="test_"
        )
        u_sol = problem.solve()
        
        # Sample on 50x50 grid
        nx = ny = 50
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx*ny)]).T
        
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
        
        if np.any(np.isnan(u_values)):
            u_values_all = comm.allgather(u_values)
            u_values = u_values_all[0]
        
        u_grid = u_values.reshape(nx, ny)
        
        # Load reference
        ref = np.load('oracle_output/reference.npz')
        u_ref = ref['u_star']
        max_diff = np.max(np.abs(u_grid - u_ref))
        end = time.time()
        print(f"N={N}, degree={degree}, max diff={max_diff:.2e}, time={end-start:.3f}s")
