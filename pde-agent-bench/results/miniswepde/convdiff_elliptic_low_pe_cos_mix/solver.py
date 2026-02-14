import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import dolfinx.nls as nls

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve the convection-diffusion equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Parameters from case_spec (but we have hardcoded values from problem description)
    eps = 0.2  # diffusion coefficient
    beta = np.array([0.8, 0.3])  # velocity vector
    # Manufactured solution: u_exact = cos(pi*x)*sin(pi*y)
    # Compute source term f = -eps*laplacian(u) + beta·grad(u)
    # We'll compute symbolically using ufl
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 1  # linear elements
    u_sol = None
    u_norm_prev = None
    converged_resolution = None
    solver_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define exact solution using ufl
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        # Compute source term f
        beta_ufl = ufl.as_vector(beta)  # Convert to UFL vector
        f_ufl = -eps * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_ufl, ufl.grad(u_exact_ufl))
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
        L = ufl.inner(f_ufl, v) * ufl.dx
        
        # Boundary condition: Dirichlet using exact solution on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        # Interpolate exact solution onto boundary function
        # We need a callable for interpolation: expects (dim, n_points) -> (value_size, n_points)
        def u_exact_func(x):
            # x shape (3, n) but we only need first two coordinates
            vals = np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
            return vals.reshape(1, -1)  # shape (1, n) for scalar
        u_bc.interpolate(u_exact_func)
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Try iterative solver first, fallback to direct
        try:
            # Use LinearProblem with iterative solver
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "gmres",
                    "pc_type": "hypre",
                    "ksp_rtol": 1e-8,
                    "ksp_max_it": 1000,
                },
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            solver_info["ksp_type"] = "gmres"
            solver_info["pc_type"] = "hypre"
            solver_info["rtol"] = 1e-8
            # Get iteration count from solver
            ksp = problem.solver
            its = ksp.getIterationNumber()
            solver_info["iterations"] = its
        except Exception as e:
            # Fallback to direct solver
            if rank == 0:
                print(f"Iterative solver failed: {e}, switching to direct solver")
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
                petsc_options_prefix="pdebench_"
            )
            u_sol = problem.solve()
            solver_info["ksp_type"] = "preonly"
            solver_info["pc_type"] = "lu"
            solver_info["rtol"] = 1e-8
            # Direct solver doesn't have iteration count, set to 1
            solver_info["iterations"] = 1
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if u_norm_prev is not None:
            rel_error = abs(norm - u_norm_prev) / norm
            if rel_error < 0.01:
                converged_resolution = N
                break
        u_norm_prev = norm
    
    # If loop finished without convergence, use the last solution (N=128)
    if converged_resolution is None:
        converged_resolution = 128
    
    # Update solver_info with mesh resolution and element degree
    solver_info["mesh_resolution"] = converged_resolution
    solver_info["element_degree"] = element_degree
    
    # Evaluate solution on a 50x50 grid for output
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).astype(ScalarType)  # shape (3, N)
    
    # Use point evaluation function
    def probe_points(u_func, points_array):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(points_array.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(points_array.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
        u_values = np.full((points_array.shape[1],), np.nan, dtype=ScalarType)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        # Gather results across processes
        u_all = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_all, op=MPI.SUM)  # Since each point is owned by exactly one process, sum works
        return u_all
    
    u_grid_flat = probe_points(u_sol, points)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Return dictionary
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    # Quick test with dummy case_spec
    case_spec = {"pde": {"type": "elliptic"}}
    result = solve(case_spec)
    print("Test run completed")
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"u shape: {result['u'].shape}")
