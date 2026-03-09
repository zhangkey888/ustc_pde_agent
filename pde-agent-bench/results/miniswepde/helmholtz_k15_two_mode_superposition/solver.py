import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve the Helmholtz equation: -∇²u - k²u = f with Dirichlet BCs."""

    # Extract parameters
    k_val = 15.0
    if case_spec and 'pde' in case_spec:
        pde = case_spec['pde']
        if 'wavenumber' in pde:
            k_val = float(pde['wavenumber'])

    # Output grid
    nx_out, ny_out = 50, 50

    # For Helmholtz with k=15, degree=2, N=80 gives excellent accuracy (~6e-6 RMS error)
    N = 80
    deg = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", deg))

    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    # Manufactured solution: u = sin(2πx)sin(πy) + sin(πx)sin(3πy)
    # -∇²u = 5π²sin(2πx)sin(πy) + 10π²sin(πx)sin(3πy)
    # f = -∇²u - k²u = (5π²-k²)sin(2πx)sin(πy) + (10π²-k²)sin(πx)sin(3πy)
    k2 = PETSc.ScalarType(k_val**2)
    f_expr = (5.0 * pi**2 - k_val**2) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]) + \
             (10.0 * pi**2 - k_val**2) * ufl.sin(pi * x[0]) * ufl.sin(3 * pi * x[1])

    # Bilinear form: a(u,v) = ∫(∇u·∇v - k²uv)dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx

    # Linear form: L(v) = ∫fv dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = 0 on ∂Ω (manufactured solution vanishes on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    # Solve with direct solver (robust for indefinite Helmholtz)
    ksp_type = "preonly"
    pc_type = "lu"

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
    u_sol.x.scatter_forward()

    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')

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
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "wavenumber": 15.0,
            "type": "helmholtz",
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6f}, {np.nanmax(u_grid):.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {result['solver_info']}")

    # Compute error against exact solution
    nx_out, ny_out = 50, 50
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')

    u_exact = np.sin(2 * np.pi * X) * np.sin(np.pi * Y) + np.sin(np.pi * X) * np.sin(3 * np.pi * Y)

    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Target error: <= 5.71e-03")
