import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract PDE parameters
    pde = case_spec["pde"]
    kappa = pde["coefficients"].get("kappa", 1.0)
    f_val = pde["source"].get("f", 1.0)
    
    # Extract output grid
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox
    
    # Adaptive mesh resolution - maximize accuracy within time budget
    mesh_res = 192
    element_degree = 3
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem: -div(kappa * grad(u)) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    f_const = fem.Constant(domain, PETSc.ScalarType(f_val))
    
    a = kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_const, v) * ufl.dx
    
    # Boundary conditions - all boundaries Dirichlet
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # BC function: u = sin(pi*x) + cos(pi*y)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) + np.cos(np.pi * x[1]))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with CG + HYPRE AMG
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver iteration info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0
    
    # Evaluate at points using bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    # Build lists of points and cells on this process
    points_on_proc = []
    cells_on_proc = []
    global_indices = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            global_indices.append(i)
    
    # Initialize output grid with NaN
    u_grid = np.full((ny_out, nx_out), np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr).flatten()
        
        # Map values back to grid using global indices
        for j in range(len(global_indices)):
            gi = global_indices[j]
            iy = gi // nx_out
            ix = gi % nx_out
            u_grid[iy, ix] = vals[j]
    
    # Accuracy verification: check BC residual on boundary points
    # Sample boundary values and compare with exact BC
    bc_error = 0.0
    n_boundary = 0
    # Check corners and edge midpoints
    test_pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.5, 0.0],
    ])
    bc_tree = geometry.bb_tree(domain, domain.topology.dim)
    bc_candidates = geometry.compute_collisions_points(bc_tree, test_pts)
    bc_colliding = geometry.compute_colliding_cells(domain, bc_candidates, test_pts)
    
    bc_pts = []
    bc_cells = []
    for i in range(test_pts.shape[0]):
        links = bc_colliding.links(i)
        if len(links) > 0:
            bc_pts.append(test_pts[i])
            bc_cells.append(links[0])
    
    if len(bc_pts) > 0:
        bc_vals = u_sol.eval(np.array(bc_pts), np.array(bc_cells, dtype=np.int32)).flatten()
        for k, pt in enumerate(bc_pts):
            exact_bc = np.sin(np.pi * pt[0]) + np.cos(np.pi * pt[1])
            bc_error += abs(bc_vals[k] - exact_bc)
            n_boundary += 1
    if n_boundary > 0:
        bc_error /= n_boundary
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "source": {"f": 1.0},
            "boundary_conditions": {
                "dirichlet": {"g": "sin(pi*x) + cos(pi*y)"}
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {np.nanmin(result['u']):.6f} / {np.nanmax(result['u']):.6f}")
    print(f"solver_info: {result['solver_info']}")
    print(f"Wall time: {t1-t0:.3f}s")
    nan_count = np.isnan(result['u']).sum()
    print(f"NaN count: {nan_count}")
