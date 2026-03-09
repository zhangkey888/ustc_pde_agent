import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    start_time = time.time()

    # Material parameters
    E = 1.0
    nu_val = 0.3
    mu = E / (2.0 * (1.0 + nu_val))
    lmbda = E * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

    comm = MPI.COMM_WORLD

    # Optimal parameters: degree 3 on 48x48 quad mesh gives error ~1.9e-7 in ~0.9s
    element_degree = 3
    N = 48

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)

    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", element_degree, (gdim,)))

    # Define manufactured solution via UFL
    x = ufl.SpatialCoordinate(domain)

    u_exact_expr = ufl.as_vector([
        ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1]),
        ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    ])

    # Strain and stress
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

    # Source term from manufactured solution
    f = -ufl.div(sigma(u_exact_expr))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Bilinear and linear forms
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.vstack([
        np.sin(2 * np.pi * x[0]) * np.cos(3 * np.pi * x[1]),
        np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    ]))

    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Solve with CG + AMG
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12

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

    iterations = problem.solver.getIterationNumber()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    n_points = points_3d.shape[1]
    u_values = np.full((n_points, gdim), np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            u_values[i, :] = vals[idx, :gdim]

    # Compute displacement magnitude
    disp_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = disp_mag.reshape((nx_eval, ny_eval))

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
    t0 = time.time()
    result = solve({})
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")

    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = np.sin(2 * np.pi * XX) * np.cos(3 * np.pi * YY)
    uy_exact = np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

    rms_error = np.sqrt(np.mean((result['u'] - mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - mag_exact))
    l2_rel = np.sqrt(np.sum((result['u'] - mag_exact)**2) / np.sum(mag_exact**2))
    print(f"RMS error: {rms_error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"L2 relative error: {l2_rel:.2e}")
