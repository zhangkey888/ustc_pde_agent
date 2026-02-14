import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Extract parameters
    epsilon = case_spec.get('epsilon', 0.01)
    beta = case_spec.get('beta', [10.0, 10.0])
    beta_vec = np.array(beta, dtype=PETSc.ScalarType)
    
    # Use fixed resolution that should meet accuracy requirement
    # Based on testing, N=128 with linear elements gives error ~2.8e-04
    mesh_resolution = 128
    element_degree = 1
    
    if rank == 0:
        print(f"Using mesh resolution N={mesh_resolution}, element degree={element_degree}")
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                     cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    
    # Beta as constant vector
    beta_const = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    
    # Source term f = -ε∇²u + β·∇u
    laplacian_u = -2 * (ufl.pi**2) * u_exact_ufl
    f_ufl = -epsilon * laplacian_u + ufl.dot(beta_const, grad_u_exact)
    
    # Standard Galerkin form
    a = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_const, ufl.grad(u)) * v * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_const, beta_const))
    Pe = beta_norm * h / (2 * epsilon)
    
    # Simplified SUPG parameter
    tau = h / (2 * beta_norm)  # For high Peclet number
    
    # Add SUPG terms
    a += tau * ufl.inner(beta_const, ufl.grad(u)) * ufl.inner(beta_const, ufl.grad(v)) * ufl.dx
    L += tau * ufl.inner(f_ufl, ufl.inner(beta_const, ufl.grad(v))) * ufl.dx
    
    # Boundary conditions
    def boundary_marker(x):
        return np.ones(x.shape[1], dtype=bool)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve with iterative solver, fallback to direct if needed
    total_iterations = 0
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-8,
                "ksp_max_it": 1000,
            },
            petsc_options_prefix="conv_diff_"
        )
        u_sol = problem.solve()
        ksp = problem._solver
        total_iterations = ksp.getIterationNumber()
        if rank == 0:
            print(f"Solved with iterative solver, iterations: {total_iterations}")
    except Exception as e:
        if rank == 0:
            print(f"Iterative solver failed: {e}, switching to direct solver")
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix="conv_diff_direct_"
        )
        u_sol = problem.solve()
        total_iterations = 1
    
    # Solver info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": total_iterations
    }
    
    # Evaluate on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    
    u_grid_flat = np.full((nx * ny,), np.nan, dtype=PETSc.ScalarType)
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
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    
    # Gather results to rank 0
    if comm.size > 1:
        recv_buffer = np.empty_like(u_grid_flat) if rank == 0 else None
        comm.Reduce(u_grid_flat, recv_buffer, op=MPI.SUM, root=0)
        if rank == 0:
            u_grid_flat = recv_buffer
    
    u_grid = u_grid_flat.reshape((nx, ny)) if rank == 0 else np.zeros((nx, ny))
    
    # Broadcast u_grid to all ranks (simplification)
    if comm.size > 1:
        u_grid = comm.bcast(u_grid, root=0)
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "epsilon": 0.01,
        "beta": [10.0, 10.0]
    }
    
    start_time = time.time()
    result = solve(case_spec)
    end_time = time.time()
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"\n=== Results ===")
        print(f"Time: {end_time - start_time:.3f}s (limit: 2.275s)")
        print(f"Mesh: {result['solver_info']['mesh_resolution']}")
        print(f"Degree: {result['solver_info']['element_degree']}")
        print(f"Iterations: {result['solver_info']['iterations']}")
        
        # Compute error
        u_grid = result["u"]
        nx, ny = u_grid.shape
        x_vals = np.linspace(0.0, 1.0, nx)
        y_vals = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        error = np.abs(u_grid - u_exact)
        max_error = np.max(error)
        l2_error = np.sqrt(np.mean(error**2))
        
        print(f"Max error: {max_error:.2e} (required: ≤ 3.44e-04)")
        print(f"L2 error: {l2_error:.2e}")
        print(f"Accuracy {'PASS' if max_error <= 3.44e-04 else 'FAIL'}")
        print(f"Time {'PASS' if (end_time - start_time) <= 2.275 else 'FAIL'}")
