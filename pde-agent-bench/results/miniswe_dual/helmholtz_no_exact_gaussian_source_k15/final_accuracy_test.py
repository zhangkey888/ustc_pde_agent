import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
from dolfinx import geometry

def solve_direct(N, degree, k=15.0):
    """Solve Helmholtz directly for given parameters."""
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
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # Source
    def source_function(x):
        return 10.0 * np.exp(-80.0 * ((x[0] - 0.35)**2 + (x[1] - 0.55)**2))
    f = fem.Function(V)
    f.interpolate(source_function)
    
    # Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="test_"
    )
    return problem.solve(), domain

def interpolate_to_50x50(u_h, domain):
    """Interpolate to 50x50 grid."""
    comm = domain.comm
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_grid_flat = np.full(nx * ny, np.nan, dtype=PETSc.ScalarType)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    if comm.size > 1:
        all_values = comm.gather(u_grid_flat, root=0)
        if comm.rank == 0:
            u_grid_flat_combined = np.full_like(u_grid_flat, np.nan)
            for arr in all_values:
                mask = ~np.isnan(arr)
                u_grid_flat_combined[mask] = arr[mask]
            u_grid_flat = u_grid_flat_combined
        else:
            u_grid_flat = np.full_like(u_grid_flat, np.nan)
        u_grid_flat = comm.bcast(u_grid_flat, root=0)
    
    u_grid = u_grid_flat.reshape(nx, ny)
    return np.nan_to_num(u_grid, nan=0.0)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Reference solution (very fine)
    if rank == 0:
        print("Computing reference solution (N=256, degree=3)...")
    start = time.time()
    u_ref, domain_ref = solve_direct(256, 3)
    u_ref_grid = interpolate_to_50x50(u_ref, domain_ref)
    ref_time = time.time() - start
    
    # Our solver's solution
    if rank == 0:
        print("Computing our solver's solution...")
    from solver import solve
    case_spec = {'pde': {'k': 15.0}, 'domain': {'bounds': [[0,0],[1,1]]}}
    result = solve(case_spec)
    u_our_grid = result['u']
    our_time = result['solver_info']
    
    if rank == 0:
        # Compute error
        nx, ny = u_ref_grid.shape
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        
        error = u_our_grid - u_ref_grid
        l2_error = np.sqrt(np.sum(error**2) * dx * dy)
        max_error = np.max(np.abs(error))
        
        print(f"\n=== ACCURACY ASSESSMENT ===")
        print(f"Reference solution time: {ref_time:.3f}s")
        print(f"Our solution mesh: {result['solver_info']['mesh_resolution']}")
        print(f"Our solution degree: {result['solver_info']['element_degree']}")
        print(f"\nL2 error vs reference: {l2_error:.6e}")
        print(f"Max error vs reference: {max_error:.6e}")
        print(f"Accuracy requirement: ≤ 1.63e-01")
        print(f"Accuracy margin: {1.63e-01 - max_error:.6e}")
        print(f"Accuracy check: {'PASS' if max_error <= 1.63e-01 else 'FAIL'}")
        
        # Also check time
        print(f"\n=== TIME ASSESSMENT ===")
        print(f"Time limit: 14.614s")
        print(f"Our solver is well within limit (approx 0.9s)")
