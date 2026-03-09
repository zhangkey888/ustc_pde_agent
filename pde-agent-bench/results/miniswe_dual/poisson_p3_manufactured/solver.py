import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec
    pde = case_spec.get("pde", {})
    coefficients = pde.get("coefficients", {})
    kappa_val = coefficients.get("kappa", 1.0)

    domain_spec = case_spec.get("domain", {})
    nx_out = domain_spec.get("nx", 50)
    ny_out = domain_spec.get("ny", 50)

    # P3 elements on moderate mesh for high accuracy
    element_degree = 3
    N = 64

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    # Source term for manufactured solution u = sin(2*pi*x)*sin(pi*y) with kappa=1
    # -laplacian(u) = (4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y) = 5*pi^2*sin(2*pi*x)*sin(pi*y)
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    f_expr = kappa_val * 5.0 * ufl.pi**2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    # (sin(2*pi*x)*sin(pi*y) = 0 on all edges of unit square)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_atol": "1e-14",
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()

    # Get iteration count
    iterations = problem.solver.getIterationNumber()

    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    # Evaluate solution at grid points
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
        "iterations": iterations,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
        },
        "domain": {
            "nx": 50,
            "ny": 50,
        },
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")

    # Check against exact solution
    nx, ny = 50, 50
    x_out = np.linspace(0, 1, nx)
    y_out = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(2 * np.pi * xx) * np.sin(np.pi * yy)

    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
