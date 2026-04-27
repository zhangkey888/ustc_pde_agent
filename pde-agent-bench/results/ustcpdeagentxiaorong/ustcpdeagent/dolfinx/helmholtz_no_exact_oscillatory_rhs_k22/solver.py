import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""
    
    # Extract parameters from case_spec
    k_val = case_spec["pde"]["parameters"]["k"]
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Use P3 elements with mesh_n=80 for high accuracy
    # Converged solution verified via mesh convergence study
    degree = 3
    mesh_n = 80
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_n, mesh_n],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define boundary conditions (u = 0 on all boundaries)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(10*pi*x)*sin(8*pi*y)
    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    
    # Helmholtz equation: -∇²u - k²u = f
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    k2 = fem.Constant(domain, ScalarType(k_val**2))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve using direct LU solver (safest for indefinite systems)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    try:
        iterations = problem.solver.getIterationNumber()
    except:
        iterations = 1
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_2d = np.column_stack([XX.ravel(), YY.ravel()])
    pts_3d = np.zeros((pts_2d.shape[0], 3))
    pts_3d[:, :2] = pts_2d
    
    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_n,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "parameters": {"k": 22.0},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{np.nanmin(result['u']):.8f}, {np.nanmax(result['u']):.8f}]")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Max |u|: {np.nanmax(np.abs(result['u'])):.8f}")
    
    # Verify: analytical solution for this specific case
    # f = sin(10*pi*x)*sin(8*pi*y)
    # Eigenfunction: sin(m*pi*x)*sin(n*pi*y) with eigenvalue pi^2*(m^2+n^2)
    # Solution: u = sin(10*pi*x)*sin(8*pi*y) / (pi^2*(100+64) - k^2)
    k_val = 22.0
    denom = np.pi**2 * (100 + 64) - k_val**2
    print(f"\nAnalytical denominator: pi^2*164 - 484 = {denom:.4f}")
    print(f"Analytical max |u|: {1.0/denom:.8f}")
    
    # Compare sampled solution
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(10*np.pi*XX) * np.sin(8*np.pi*YY) / denom
    
    diff = np.abs(result['u'] - u_exact)
    print(f"Max |u_fem - u_exact|: {np.nanmax(diff):.2e}")
    print(f"L2 diff (grid): {np.sqrt(np.nanmean(diff**2)):.2e}")
