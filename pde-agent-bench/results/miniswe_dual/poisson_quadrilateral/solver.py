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
    kappa_val = 2.0
    
    # Parse case_spec if available
    if case_spec and 'pde' in case_spec:
        pde = case_spec['pde']
        if 'coefficients' in pde:
            kappa_val = pde['coefficients'].get('kappa', kappa_val)
    
    # Output grid
    nx_out, ny_out = 50, 50
    
    # Adaptive mesh refinement
    # For quadrilateral elements with degree 2, we need sufficient resolution
    # Target error: 8.70e-05, so we need good accuracy
    element_degree = 2
    N = 64  # Start with moderate resolution for quads with degree 2
    
    # Create mesh with quadrilateral cells
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    # Function space - Lagrange on quads
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define exact solution symbolically
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact_expr = ufl.exp(x[0]) * ufl.cos(2 * pi * x[1])
    
    # Compute source term: f = -div(kappa * grad(u_exact))
    # For u = exp(x)*cos(2*pi*y), kappa = 2:
    # grad(u) = (exp(x)*cos(2*pi*y), -2*pi*exp(x)*sin(2*pi*y))
    # div(kappa*grad(u)) = kappa * (exp(x)*cos(2*pi*y) + exp(x)*(-4*pi^2)*cos(2*pi*y))
    #                     = kappa * exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    # f = -div(kappa*grad(u)) = -kappa * exp(x)*cos(2*pi*y)*(1 - 4*pi^2)
    #                         = kappa * exp(x)*cos(2*pi*y)*(4*pi^2 - 1)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_expr))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions - u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Create BC function by interpolating exact solution
    u_bc = fem.Function(V)
    u_exact_fe = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_exact_fe)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
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
    
    # Evaluate solution on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    
    # Create 3D points array (required by dolfinx even for 2D)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = xx.flatten()
    points[1, :] = yy.flatten()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
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
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    
    u_exact = np.exp(xx) * np.cos(2 * np.pi * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
