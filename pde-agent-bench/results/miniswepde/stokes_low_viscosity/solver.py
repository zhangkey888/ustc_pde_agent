import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nu_val = 0.1
    N = 96
    degree_u = 2
    degree_p = 1

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Create elements
    e_u = basix.ufl.element("Lagrange", "triangle", degree_u, shape=(domain.geometry.dim,))
    e_p = basix.ufl.element("Lagrange", "triangle", degree_p)
    mel = basix.ufl.mixed_element([e_u, e_p])

    # Mixed function space
    W = fem.functionspace(domain, mel)

    # Also create individual spaces for BCs and post-processing
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_s = ufl.pi

    # Exact solution (UFL)
    u_exact = ufl.as_vector([pi_s * ufl.cos(pi_s * x[1]) * ufl.sin(pi_s * x[0]),
                              -pi_s * ufl.cos(pi_s * x[0]) * ufl.sin(pi_s * x[1])])
    p_exact = ufl.cos(pi_s * x[0]) * ufl.cos(pi_s * x[1])

    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Bilinear form
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions - all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    # Velocity BC
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()

    # Extract velocity
    u_h = wh.sub(0).collapse()

    # Evaluate on 100x100 grid
    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    n_points = points_3d.shape[1]
    vel_mag = np.full(n_points, np.nan)

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
        vals = u_h.eval(pts_arr, cells_arr)
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]

    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)

    error = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    rel_error = error / np.sqrt(np.nanmean(vel_mag_exact**2))
    max_error = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    print(f"RMS error: {error:.2e}")
    print(f"Relative RMS error: {rel_error:.2e}")
    print(f"Max abs error: {max_error:.2e}")
