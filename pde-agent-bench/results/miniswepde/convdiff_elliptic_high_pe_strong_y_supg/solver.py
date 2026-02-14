import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization.
    Adaptive mesh refinement loop with convergence check.
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    epsilon = case_spec.get('epsilon', 0.01)
    beta = case_spec.get('beta', [0.0, 15.0])
    beta_array = np.array(beta, dtype=ScalarType)
    
    # Adaptive loop
    resolutions = [32, 64, 128]
    u_sol = None
    norm_old = None
    mesh_resolution_used = resolutions[-1]  # fallback
    degree = 1
    total_iterations = 0
    ksp_type_used = "gmres"
    pc_type_used = "hypre"
    domain = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, nx=N, ny=N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", degree))
        
        # Beta constant
        beta_const = fem.Constant(domain, beta_array)
        
        # Manufactured solution
        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        
        # Compute f from PDE: -ε∇²u + β·∇u = f
        f_expr = -epsilon * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_const, ufl.grad(u_exact))
        
        # Boundary condition (Dirichlet, from exact solution)
        def boundary_marker(x):
            return np.ones(x.shape[1], dtype=bool)  # All boundaries
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        u_bc = fem.Function(V)
        # Interpolation function must return shape (num_points,) for scalar
        u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # SUPG parameter (optimal for linear elements)
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta_const, beta_const))
        # Local Péclet number
        Pe = beta_norm * h / (2 * epsilon + 1e-14)
        # Optimal xi = coth(Pe) - 1/Pe
        xi = ufl.conditional(Pe > 1e-12, 1 / ufl.tanh(Pe) - 1 / Pe, Pe / 3)  # Taylor expansion for small Pe
        tau = h / (2 * beta_norm) * xi
        
        # Bilinear form with SUPG
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.inner(ufl.dot(beta_const, ufl.grad(u)), v)) * ufl.dx
        a += tau * ufl.inner(ufl.dot(beta_const, ufl.grad(u)), ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Linear form
        L = ufl.inner(f_expr, v) * ufl.dx
        L += tau * ufl.inner(f_expr, ufl.dot(beta_const, ufl.grad(v))) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
                petsc_options_prefix="conv_"
            )
            u_sol = problem.solve()
            # Get iteration count
            ksp = problem.solver
            total_iterations += ksp.getIterationNumber()
            ksp_type_used = "gmres"
            pc_type_used = "hypre"
        except Exception as e:
            # Fallback to direct solver
            print(f"Iterative solver failed at N={N}, falling back to direct: {e}")
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-8},
                petsc_options_prefix="conv_"
            )
            u_sol = problem.solve()
            ksp = problem.solver
            total_iterations += ksp.getIterationNumber()
            ksp_type_used = "preonly"
            pc_type_used = "lu"
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_value = np.sqrt(comm.allreduce(fem.assemble_scalar(norm_form), op=MPI.SUM))
        
        # Check convergence
        if norm_old is not None:
            relative_error = abs(norm_value - norm_old) / (norm_value + 1e-14)
            if relative_error < 0.01:  # 1% convergence
                mesh_resolution_used = N
                break
        
        norm_old = norm_value
    
    # Evaluate on 50x50 grid
    nx = ny = 50
    x_vals = np.linspace(0, 1, nx)
    y_vals = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx * ny)]).T
    
    # Probe points
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
    
    u_values = np.full((points.shape[0],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(nx, ny)
    
    # Solver info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": degree,
        "ksp_type": ksp_type_used,
        "pc_type": pc_type_used,
        "rtol": 1e-8,
        "iterations": total_iterations
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    # Test with given case spec
    case_spec = {
        "epsilon": 0.01,
        "beta": [0.0, 15.0]
    }
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)
