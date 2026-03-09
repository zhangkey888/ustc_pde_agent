import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    comm = MPI.COMM_WORLD
    
    # Parameters - need higher resolution due to exp(6*y) boundary layer
    element_degree = 2
    N = 80
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Coefficient kappa = 1.0
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # Exact (manufactured) solution: u = exp(6*y)*sin(pi*x)
    u_exact_ufl = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    
    # Source term f = -div(kappa * grad(u_exact))
    # For kappa=1: -laplacian(u) = -(d^2u/dx^2 + d^2u/dy^2)
    # d^2u/dx^2 = -pi^2 * exp(6y) * sin(pi*x)
    # d^2u/dy^2 = 36 * exp(6y) * sin(pi*x)
    # -laplacian(u) = (pi^2 - 36) * exp(6y) * sin(pi*x)
    # So f = (pi^2 - 36) * exp(6y) * sin(pi*x)
    f = -ufl.div(kappa * ufl.grad(u_exact_ufl))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Solve with CG + Hypre (AMG)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {}
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Time: {elapsed:.3f}s")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    
    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(6.0 * YY) * np.sin(np.pi * XX)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Solver info: {result['solver_info']}")
    
    print(f"\n--- Pass/Fail ---")
    print(f"Accuracy target: 4.40e-04")
    print(f"Time target: 2.143 s")
    print(f"RMS error: {error:.6e} {'PASS' if error < 4.4e-4 else 'FAIL'}")
    print(f"Wall time: {elapsed:.3f} s {'PASS' if elapsed < 2.143 else 'FAIL'}")
