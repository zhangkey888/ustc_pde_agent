import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse case spec
    kappa_val = 5.0
    if 'pde' in case_spec and 'coefficients' in case_spec['pde']:
        coeffs = case_spec['pde']['coefficients']
        if 'kappa' in coeffs:
            kappa_val = float(coeffs['kappa'])
    
    # Get output grid size
    nx_out = 50
    ny_out = 50
    if 'output' in case_spec:
        nx_out = case_spec['output'].get('nx', 50)
        ny_out = case_spec['output'].get('ny', 50)
    
    # Adaptive mesh refinement
    comm = MPI.COMM_WORLD
    
    # For this problem with cos(2πx)cos(3πy), degree 2 elements on moderate mesh should suffice
    N = 64
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    
    # Source term: f = kappa * 13 * pi^2 * cos(2*pi*x)*cos(3*pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.cos(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    f_expr = kappa_val * 13.0 * (ufl.pi**2) * ufl.cos(2 * pi * x[0]) * ufl.cos(3 * pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = u_exact on boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(3 * np.pi * x[1]))
    
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
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 5.0},
        },
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    u_exact = np.cos(2 * np.pi * xx) * np.cos(3 * np.pi * yy)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
