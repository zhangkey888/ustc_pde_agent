import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation with manufactured solution."""
    
    comm = MPI.COMM_WORLD
    
    # Adaptive strategy: try increasing resolution until convergence
    # For this high-frequency problem, use degree 2 elements for efficiency
    element_degree = 2
    N = 64  # Start with moderate resolution
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution (manufactured)
    u_exact_expr = ufl.sin(3 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    
    # Coefficient
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Source term: f = -div(kappa * grad(u_exact))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_expr))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω (g = u_exact)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Interpolate exact solution for BC
    u_bc_func = fem.Function(V)
    u_exact_expression = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expression)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    # Point evaluation
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
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution on the grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.sin(3 * np.pi * (XX + YY)) * np.sin(np.pi * (XX - YY))
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
