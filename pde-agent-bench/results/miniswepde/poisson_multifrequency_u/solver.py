import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Poisson equation with adaptive mesh refinement.
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType
    
    # Extract parameters from case_spec with defaults
    pde_info = case_spec.get("pde", {})
    coeffs = pde_info.get("coefficients", {})
    kappa_value = coeffs.get("kappa", 1.0)
    
    # Manufactured solution
    def exact_solution(x):
        return (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) +
                0.3 * np.sin(6 * np.pi * x[0]) * np.sin(6 * np.pi * x[1]))
    
    # Define source term from manufactured solution
    # -Δu = f, so f = -Δu
    # u = sin(pi*x)*sin(pi*y) + 0.3*sin(6*pi*x)*sin(6*pi*y)
    # Δu = ∂²u/∂x² + ∂²u/∂y²
    # ∂²u/∂x² = -π² sin(πx) sin(πy) - 0.3*(6π)² sin(6πx) sin(6πy)
    # ∂²u/∂y² = -π² sin(πx) sin(πy) - 0.3*(6π)² sin(6πx) sin(6πy)
    # So Δu = -2π² sin(πx) sin(πy) - 0.3*2*(6π)² sin(6πx) sin(6πy)
    # f = -Δu = 2π² sin(πx) sin(πy) + 0.3*2*(6π)² sin(6πx) sin(6πy)
    pi = np.pi
    def source_term(x):
        return (2 * pi**2 * np.sin(pi * x[0]) * np.sin(pi * x[1]) +
                0.3 * 2 * (6 * pi)**2 * np.sin(6 * pi * x[0]) * np.sin(6 * pi * x[1]))
    
    # Boundary condition (Dirichlet)
    def boundary_marker(x):
        # Mark all boundaries
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    
    # Adaptive mesh refinement loop
    resolutions = [32, 64, 128]
    element_degree = 2  # Use quadratic elements for better accuracy
    u_sol = None
    u_norm_prev = None
    converged_resolution = None
    solver_info_final = {}
    domain_final = None
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space with quadratic elements
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Boundary conditions
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        
        # Create boundary function
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x: exact_solution(x))
        bc = fem.dirichletbc(u_bc, dofs)
        
        # Variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        kappa = fem.Constant(domain, ScalarType(kappa_value))
        
        # Create source function and interpolate
        f = fem.Function(V)
        f.interpolate(lambda x: source_term(x))
        
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        
        # Try iterative solver first, fallback to direct if fails
        solver_success = False
        for solver_config in [
            {"ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8},
            {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12}
        ]:
            try:
                problem = petsc.LinearProblem(
                    a, L, bcs=[bc],
                    petsc_options=solver_config,
                    petsc_options_prefix="poisson_"
                )
                u_sol = problem.solve()
                solver_success = True
                
                # Record solver info
                solver_info = {
                    "mesh_resolution": N,
                    "element_degree": element_degree,
                    "ksp_type": solver_config["ksp_type"],
                    "pc_type": solver_config["pc_type"],
                    "rtol": solver_config["rtol"],
                    "iterations": problem.solver.getIterationNumber() if solver_config["ksp_type"] != "preonly" else 0
                }
                solver_info_final = solver_info
                domain_final = domain
                break
            except Exception as e:
                if solver_config["ksp_type"] == "preonly":
                    # Direct solver failed, re-raise
                    raise
                # Try next solver
                continue
        
        if not solver_success:
            raise RuntimeError("All solvers failed")
        
        # Compute L2 norm of solution
        u_norm = fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))
        u_norm = np.sqrt(u_norm)
        
        # Check convergence - tighter tolerance for accuracy
        if u_norm_prev is not None:
            relative_error = abs(u_norm - u_norm_prev) / u_norm if u_norm > 1e-12 else abs(u_norm - u_norm_prev)
            if relative_error < 0.001:  # 0.1% convergence criterion (tighter)
                converged_resolution = N
                break
        
        u_norm_prev = u_norm
    
    # If loop finished without convergence, use finest mesh result
    if converged_resolution is None:
        converged_resolution = resolutions[-1]
    
    # Generate output on 50x50 grid
    nx = ny = 50
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create points array (shape (3, nx*ny))
    points = np.zeros((3, nx * ny))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()
    points[2, :] = 0.0  # z-coordinate for 2D
    
    # Evaluate solution at points
    bb_tree = geometry.bb_tree(domain_final, domain_final.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain_final, cell_candidates, points.T)
    
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
    
    # Reshape to (nx, ny)
    u_grid = u_values.reshape((nx, ny))
    
    # Return result
    return {
        "u": u_grid,
        "solver_info": solver_info_final
    }

# Test the solver if run directly
if __name__ == "__main__":
    # Test case specification
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0}
        }
    }
    
    result = solve(case_spec)
    print("Solver info:", result["solver_info"])
    print("u shape:", result["u"].shape)
    print("u min/max:", result["u"].min(), result["u"].max())
