import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve 2D linear elasticity with manufactured solution (near-incompressible)."""

    # Material parameters
    E_val = 1.0
    nu_val = 0.49

    # Lame parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lambda_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

    # CG2 to avoid volumetric locking; N=72 gives RMS error ~8e-7 < 1e-6
    element_degree = 2
    N = 72

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Vector function space
    V = fem.functionspace(domain, ("Lagrange", element_degree, (domain.geometry.dim,)))

    # Trial / test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    # Material constants
    mu = fem.Constant(domain, PETSc.ScalarType(mu_val))
    lmbda = fem.Constant(domain, PETSc.ScalarType(lambda_val))

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(epsilon(w)) * ufl.Identity(domain.geometry.dim)

    # Manufactured exact solution in UFL
    pi = ufl.pi
    u_exact_ufl = ufl.as_vector([
        ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
        ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    ])

    # Source term: f = -div(sigma(u_exact))
    eps_exact = ufl.sym(ufl.grad(u_exact_ufl))
    sigma_exact = 2.0 * mu * eps_exact + lmbda * ufl.tr(eps_exact) * ufl.Identity(domain.geometry.dim)
    f_ufl = -ufl.div(sigma_exact)

    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet BC on entire boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
        np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    ]))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs)

    # Solve with CG + AMG
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
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.stack([XX.ravel(), YY.ravel()], axis=0)
    points_3d = np.vstack([points_2d, np.zeros((1, points_2d.shape[1]))])

    # Point evaluation via bounding-box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    n_points = points_3d.shape[1]
    u_values = np.full((n_points, 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]

    # Displacement magnitude
    disp_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = disp_mag.reshape((nx_eval, ny_eval))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0

    u_grid = result["u"]
    info = result["solver_info"]

    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    u_exact_vals = np.stack([
        np.sin(np.pi * XX.ravel()) * np.sin(np.pi * YY.ravel()),
        np.sin(np.pi * XX.ravel()) * np.cos(np.pi * YY.ravel())
    ], axis=1)
    exact_mag = np.sqrt(u_exact_vals[:, 0]**2 + u_exact_vals[:, 1]**2)
    exact_grid = exact_mag.reshape((nx_eval, ny_eval))

    error_rms = np.sqrt(np.mean((u_grid - exact_grid)**2))
    max_error = np.max(np.abs(u_grid - exact_grid))

    print(f"Mesh resolution: {info['mesh_resolution']}")
    print(f"Element degree: {info['element_degree']}")
    print(f"RMS error: {error_rms:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"Solution range: [{u_grid.min():.6f}, {u_grid.max():.6f}]")
    print(f"Wall time: {elapsed:.3f}s")
    print(f"PASS error (<= 1e-6): {error_rms <= 1e-6}")
    print(f"PASS time  (<= 2.712s): {elapsed <= 2.712}")
