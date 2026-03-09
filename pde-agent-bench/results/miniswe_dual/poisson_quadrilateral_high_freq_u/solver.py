import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa_val = float(coefficients.get("kappa", 1.0))
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # For sin(4*pi*x)*sin(4*pi*y), we need good resolution
    # With degree 2 elements, N=64 should be sufficient for ~1e-3 error
    # Let's use adaptive approach
    
    N = 48
    degree = 2
    ksp_type_used = "cg"
    pc_type_used = "hypre"
    rtol_used = 1e-10
    total_iterations = 0
    
    # Create mesh - quadrilateral as specified in case ID
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    
    # Source term: -div(kappa * grad(u_exact)) = kappa * 32 * pi^2 * sin(4*pi*x)*sin(4*pi*y)
    f_ufl = -kappa_val * ufl.div(ufl.grad(u_exact_ufl))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    a = kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Interpolate exact solution for BC
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "ksp_rtol": str(rtol_used),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    try:
        total_iterations = problem.solver.getIterationNumber()
    except:
        total_iterations = 0
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": total_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0},
        },
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Iterations: {result['solver_info']['iterations']}")
    
    # Compute error against exact solution
    nx, ny = u_grid.shape
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
