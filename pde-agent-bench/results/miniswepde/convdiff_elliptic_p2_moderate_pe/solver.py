import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    """
    Solve convection-diffusion equation with SUPG stabilization.
    Adaptive mesh refinement based on convergence of solution norm.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Parameters from case_spec
    epsilon = case_spec.get('epsilon', 0.03)
    beta = case_spec.get('beta', [5.0, 2.0])
    beta_vec = np.array(beta, dtype=ScalarType)
    
    # Manufactured solution
    def u_exact_func(x):
        return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    # Source term f derived from manufactured solution
    # -ε∇²u + β·∇u = f
    # Compute f analytically
    x = ufl.SpatialCoordinate(ufl.triangle)  # placeholder, will be replaced per mesh
    
    # Define exact solution as UFL expression for computing f
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_ufl = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_vec, ufl.grad(u_exact_ufl))
    
    # Grid convergence loop
    resolutions = [32, 64, 128]
    element_degree = 2  # P2 elements for better accuracy
    u_sol = None
    norm_old = None
    mesh_resolution_used = None
    solver_info = {}
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define exact solution and source term on this mesh
        x = ufl.SpatialCoordinate(domain)
        u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
        f_ufl = -epsilon * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta_vec, ufl.grad(u_exact_ufl))
        
        # Interpolate exact solution for boundary condition
        u_exact = fem.Function(V)
        u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1]))
        
        # Dirichlet BC on entire boundary
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(u_exact, dofs)
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # SUPG parameter (tau)
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(beta_vec[0]**2 + beta_vec[1]**2)
        Pe = beta_norm * h / (2 * epsilon)  # local Péclet number
        tau = h / (2 * beta_norm) * (1 / ufl.tanh(Pe) - 1 / Pe)
        
        # Bilinear form with SUPG
        a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) 
             + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v)
             + tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v)))) * ufl.dx
        
        # Linear form with SUPG
        f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
        f_func = fem.Function(V)
        f_func.interpolate(f_expr)
        L = (ufl.inner(f_func, v) 
             + tau * ufl.inner(f_func, ufl.dot(beta_vec, ufl.grad(v)))) * ufl.dx
        
        # Solve linear system
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-8},
            petsc_options_prefix="conv_"
        )
        u_sol = problem.solve()
        
        # Compute L2 norm of solution
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_local = fem.assemble_scalar(norm_form)
        norm_global = comm.allreduce(norm_local, op=MPI.SUM)
        norm_new = np.sqrt(norm_global)
        
        # Check convergence
        if norm_old is not None:
            rel_error = abs(norm_new - norm_old) / norm_new
            if rel_error < 0.01:
                mesh_resolution_used = N
                break
        norm_old = norm_new
        
        # If last resolution, use it
        if N == resolutions[-1]:
            mesh_resolution_used = N
    
    # Prepare output grid (50x50)
    nx = ny = 50
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(0.0, 1.0, ny)
    points = np.array(np.meshgrid(x_vals, y_vals, indexing='ij')).reshape(2, -1)
    points_3d = np.vstack([points, np.zeros((1, points.shape[1]))])  # make 3D
    
    # Evaluate solution at points
    u_grid_flat = evaluate_at_points(u_sol, points_3d, domain)
    u_grid = u_grid_flat.reshape((nx, ny))
    
    # Solver info
    solver_info = {
        "mesh_resolution": mesh_resolution_used,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1e-8,
        "iterations": problem.solver.getIterationNumber() if hasattr(problem.solver, 'getIterationNumber') else 0,
    }
    
    return {"u": u_grid, "solver_info": solver_info}

def evaluate_at_points(u_func, points, domain):
    """Evaluate FEM function at given points (shape (3, N))."""
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
    
    u_values = np.full((points.shape[1],), np.nan, dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    return u_values

if __name__ == "__main__":
    # Test with given parameters
    case_spec = {
        "epsilon": 0.03,
        "beta": [5.0, 2.0],
    }
    result = solve(case_spec)
    print(f"Mesh resolution used: {result['solver_info']['mesh_resolution']}")
    print(f"u shape: {result['u'].shape}")
