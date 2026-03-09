import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # For this multi-frequency problem, we need good resolution
    # The high-frequency term has 5*pi and 4*pi, so we need fine mesh + high degree
    N = 64
    degree = 3
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + 0.2 * ufl.sin(5*pi*x[0]) * ufl.sin(4*pi*x[1])
    
    # kappa = 1.0, so f = -div(kappa * grad(u)) = -laplacian(u)
    # For u = sin(pi*x)*sin(pi*y): -laplacian = 2*pi^2 * sin(pi*x)*sin(pi*y)
    # For u = 0.2*sin(5*pi*x)*sin(4*pi*y): -laplacian = 0.2*(25*pi^2 + 16*pi^2)*sin(5*pi*x)*sin(4*pi*y)
    # = 0.2*41*pi^2*sin(5*pi*x)*sin(4*pi*y)
    f_expr = 2.0 * pi**2 * ufl.sin(pi*x[0]) * ufl.sin(pi*x[1]) + \
             0.2 * 41.0 * pi**2 * ufl.sin(5*pi*x[0]) * ufl.sin(4*pi*x[1])
    
    # Boundary conditions: u = g on boundary (from exact solution)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_fem_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_exact_fem_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
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
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,  # not easily accessible from LinearProblem
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    nx, ny = u_grid.shape
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) + 0.2 * np.sin(5*np.pi*X) * np.sin(4*np.pi*Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 (RMS) error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
