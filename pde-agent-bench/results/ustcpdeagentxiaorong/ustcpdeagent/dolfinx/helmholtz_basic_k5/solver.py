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
    k_val = case_spec.get("pde", {}).get("wavenumber", 5.0)
    
    # Output grid specification
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution - use high resolution for accuracy
    # With P3 elements, N=80 should give very good accuracy
    N = 96
    degree = 3
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term: f = (2*pi^2 - k^2) * sin(pi*x) * sin(pi*y)
    k2 = fem.Constant(domain, ScalarType(k_val**2))
    f = (2.0 * ufl.pi**2 - k2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Weak form: ∫∇u·∇v dx - k²∫uv dx = ∫fv dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Dirichlet boundary conditions
    # u = sin(pi*x)*sin(pi*y) = 0 on all boundaries of [0,1]^2
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using direct solver (robust for indefinite systems)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Compute L2 error for verification
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {l2_error:.6e}")
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    # Build bounding box tree and find cells
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
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Get iteration count
    solver = problem.solver
    iterations = solver.getIterationNumber()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    import time
    
    # Test case
    case_spec = {
        "pde": {
            "wavenumber": 5.0,
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
    
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify against exact solution on the grid
    grid_spec = case_spec["output"]["grid"]
    xs = np.linspace(grid_spec["bbox"][0], grid_spec["bbox"][1], grid_spec["nx"])
    ys = np.linspace(grid_spec["bbox"][2], grid_spec["bbox"][3], grid_spec["ny"])
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    
    grid_error = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact_grid))
    print(f"Grid RMS error: {grid_error:.6e}")
    print(f"Grid max error: {max_error:.6e}")
