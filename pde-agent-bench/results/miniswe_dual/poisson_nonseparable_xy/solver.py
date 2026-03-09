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
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Adaptive mesh refinement
    # For sin(pi*x*y), we need sufficient resolution. With degree 2, N=64 should be fine.
    # But let's do a quick convergence check.
    
    N = 80
    degree = 2
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for BCs and source term
    u_exact_ufl = ufl.sin(ufl.pi * x[0] * x[1])
    
    # Compute source term: f = -div(kappa * grad(u_exact))
    # For kappa=1: f = -laplacian(u_exact)
    # u = sin(pi*x*y)
    # du/dx = pi*y*cos(pi*x*y), d2u/dx2 = -pi^2*y^2*sin(pi*x*y)
    # du/dy = pi*x*cos(pi*x*y), d2u/dy2 = -pi^2*x^2*sin(pi*x*y)
    # laplacian = -pi^2*(x^2+y^2)*sin(pi*x*y)
    # f = -kappa * laplacian = kappa * pi^2*(x^2+y^2)*sin(pi*x*y)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    f_expr = kappa * ufl.pi**2 * (x[0]**2 + x[1]**2) * ufl.sin(ufl.pi * x[0] * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundaries
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol_val),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Points array: shape (N_points, 3)
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = xx.flatten()
    points_3d[:, 1] = yy.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points_3d)):
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
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": iterations,
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
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(np.pi * xx * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
