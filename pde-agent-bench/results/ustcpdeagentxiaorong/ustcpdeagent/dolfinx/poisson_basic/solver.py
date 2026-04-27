import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid specs
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Solver parameters - high accuracy within time budget
    mesh_resolution = 100
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    # Create mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Define exact solution and source term using UFL
    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # For -∇·(κ ∇u) = f with κ=1, u=sin(πx)sin(πy):
    # f = 2π² sin(πx) sin(πy)
    f_expr = 2.0 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(domain, ScalarType(1.0))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Solve
    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
        "ksp_max_it": 2000,
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=petsc_options,
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()

    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

    # Build bounding box tree
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

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    # Test with a mock case_spec
    case_spec = {
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
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
    
    # Check against exact solution on the grid
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    grid_error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    print(f"Grid RMS error: {grid_error:.6e}")
    print(f"Max abs error on grid: {np.nanmax(np.abs(result['u'] - u_exact)):.6e}")
    
    # Check NaN values
    nan_count = np.sum(np.isnan(result['u']))
    print(f"NaN count: {nan_count}")
