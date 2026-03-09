import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa_val = float(coefficients.get("kappa", 0.1))
    
    domain_spec = case_spec.get("domain", {})
    nx_out = domain_spec.get("nx", 50)
    ny_out = domain_spec.get("ny", 50)
    
    # Adaptive mesh refinement
    comm = MPI.COMM_WORLD
    
    # For this problem with P2 elements, N=32 should be more than sufficient
    # for error < 5.81e-04. Let's use adaptive approach.
    resolutions = [32, 64]
    element_degree = 2
    
    prev_norm = None
    u_sol = None
    domain = None
    V = None
    final_N = None
    ksp_type_used = "cg"
    pc_type_used = "hypre"
    rtol_used = 1e-10
    iterations_used = 0
    
    for N in resolutions:
        # Create mesh
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        
        # Function space
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        x = ufl.SpatialCoordinate(domain)
        
        # Source term: f = kappa * 2 * pi^2 * sin(pi*x) * sin(pi*y)
        f_expr = kappa_val * 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        
        # Bilinear and linear forms
        kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa_val))
        a = kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f_expr * v * ufl.dx
        
        # Boundary conditions: u = sin(pi*x)*sin(pi*y) on boundary
        # On the unit square boundary, sin(pi*x)*sin(pi*y) = 0
        # (since either x=0,1 or y=0,1 makes sin=0)
        tdim = domain.topology.dim
        fdim = tdim - 1
        
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
        
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
        
        # Check convergence
        current_norm = np.sqrt(comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
            op=MPI.SUM
        ))
        
        final_N = N
        
        if prev_norm is not None:
            rel_change = abs(current_norm - prev_norm) / (current_norm + 1e-15)
            if rel_change < 0.001:
                break
        
        prev_norm = current_norm
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
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
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": iterations_used,
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 0.1},
        },
        "domain": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
