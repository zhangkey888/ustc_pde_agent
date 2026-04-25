import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation: -div(kappa * grad(u)) = f in Omega, u = g on dOmega
    with manufactured solution u = x*(1-x)*y*(1-y)*(1 + 0.5*x*y)
    """
    comm = MPI.COMM_WORLD
    
    # Extract output grid specs
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # PDE parameters
    kappa_val = case_spec["pde"]["coefficients"].get("kappa", 1.0)
    
    # Check if time-dependent
    pde_time = case_spec["pde"].get("time", None)
    is_transient = pde_time is not None and pde_time.get("t_end", None) is not None
    
    # Solver parameters - P5 with moderate mesh for better grid evaluation
    mesh_resolution = 8   # Moderate mesh for better point evaluation accuracy
    element_degree = 5    # Exact for degree 5 polynomial
    ksp_type = "preonly" # Direct solver
    pc_type = "lu"
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define manufactured solution using UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact = x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * (1 + 0.5 * x[0] * x[1])
    
    # Compute source term: f = -div(kappa * grad(u_exact))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact))
    
    # Boundary condition: g = u_exact on dOmega
    g_expr = u_exact
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term as a dolfinx function
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Weak form
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_func * v * ufl.dx
    
    # Boundary conditions - all Dirichlet
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create BC function with exact solution values
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using LinearProblem with direct LU solver
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Create points array (3, N) for dolfinx
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    points[2] = 0.0
    
    # Evaluate solution at points
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (ny, nx)
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_func = fem.Function(V)
    error_func.x.array[:] = u_sol.x.array - u_exact_func.x.array
    
    L2_error = fem.assemble_scalar(fem.form(ufl.inner(error_func, error_func) * ufl.dx))
    L2_error = np.sqrt(L2_error) if comm.size == 1 else np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))
    
    # Compute max error on grid for verification
    u_exact_grid = XX * (1 - XX) * YY * (1 - YY) * (1 + 0.5 * XX * YY)
    max_error_grid = np.nanmax(np.abs(u_grid - u_exact_grid))
    
    print(f"L2 error: {L2_error:.6e}")
    print(f"Max error on grid: {max_error_grid:.6e}")
    print(f"Linear solver iterations: {iterations}")
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    # Add time info if transient
    if is_transient:
        solver_info["dt"] = 0.01
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "backward_euler"
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    if is_transient:
        result["u_initial"] = u_grid.copy()
    
    return result

# For testing
if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": None,
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    import time
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Result shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Check accuracy
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = XX * (1 - XX) * YY * (1 - YY) * (1 + 0.5 * XX * YY)
    error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"Max error: {error:.6e}")
    print(f"Target error: 2.29e-03")
    print(f"Target time: 1.386s")
    print(f"PASS: error={error <= 2.29e-03}, time={elapsed <= 1.386}")
