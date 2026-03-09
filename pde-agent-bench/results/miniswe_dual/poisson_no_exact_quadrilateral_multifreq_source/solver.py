import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation with multi-frequency source."""
    
    comm = MPI.COMM_WORLD
    
    # Parse case_spec for parameters (with defaults)
    pde_spec = case_spec.get("pde", {})
    coeffs = pde_spec.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))
    
    # Domain
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    # Adaptive mesh refinement
    mesh_resolution = 64
    element_degree = 2
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-10
    
    # Create quadrilateral mesh
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Source term: f = sin(6*pi*x)*sin(5*pi*y) + 0.4*sin(11*pi*x)*sin(9*pi*y)
    f_expr = ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1]) + \
             0.4 * ufl.sin(11 * pi * x[0]) * ufl.sin(9 * pi * x[1])
    
    # Bilinear and linear forms
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions (homogeneous Dirichlet)
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
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol_val),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = 0
    try:
        ksp = problem.solver
        iterations = ksp.getIterationNumber()
    except:
        iterations = -1
    
    # Evaluate solution on output grid
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    
    # Create 3D points array for evaluation
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells for each point
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    # Evaluate
    u_grid = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(points_3d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "poisson",
            "coefficients": {"kappa": 1.0},
            "source": "sin(6*pi*x)*sin(5*pi*y) + 0.4*sin(11*pi*x)*sin(9*pi*y)",
        },
        "domain": {
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
        },
        "output": {
            "nx": 50,
            "ny": 50,
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Compute reference: for sin(nπx)sin(mπy) source, exact solution is
    # u = sin(nπx)sin(mπy) / (π²(n²+m²))
    # For f = sin(6πx)sin(5πy) + 0.4*sin(11πx)sin(9πy)
    x_coords = np.linspace(0, 1, 50)
    y_coords = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    u_exact = np.sin(6*np.pi*xx)*np.sin(5*np.pi*yy) / (np.pi**2 * (36 + 25)) + \
              0.4 * np.sin(11*np.pi*xx)*np.sin(9*np.pi*yy) / (np.pi**2 * (121 + 81))
    
    # L2 error (grid-based)
    error = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    rel_error = error / np.sqrt(np.nanmean(u_exact**2))
    print(f"L2 error (grid): {error:.6e}")
    print(f"Relative L2 error: {rel_error:.6e}")
    print(f"Max absolute error: {np.nanmax(np.abs(u_grid - u_exact)):.6e}")
