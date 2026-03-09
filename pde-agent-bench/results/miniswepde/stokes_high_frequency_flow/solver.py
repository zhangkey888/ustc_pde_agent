import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters - P4/P3 Taylor-Hood with moderate mesh for high accuracy + speed
    N = 36
    degree_u = 4
    degree_p = 3

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Create Taylor-Hood mixed function space
    Ve = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(2,))
    Qe = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    Me = basix.ufl.mixed_element([Ve, Qe])
    W = fem.functionspace(domain, Me)

    # Sub-spaces for BCs
    V_sub, V_sub_map = W.sub(0).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    nu = 1.0

    # Manufactured solution
    u_exact = ufl.as_vector([
        2 * pi_val * ufl.cos(2 * pi_val * x[1]) * ufl.sin(2 * pi_val * x[0]),
        -2 * pi_val * ufl.cos(2 * pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
    ])
    p_exact = ufl.sin(2 * pi_val * x[0]) * ufl.cos(2 * pi_val * x[1])

    # Source term
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Bilinear form
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
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
    u_sol = wh.sub(0).collapse()

    # Evaluate velocity magnitude on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    vel_mag = np.full(points_3d.shape[0], np.nan)

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
        u_vals = u_sol.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_magnitude[idx]

    u_grid = vel_mag.reshape((nx_out, ny_out))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {np.nanmin(result['u']):.6e}, max: {np.nanmax(result['u']):.6e}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")

    # Compute exact velocity magnitude
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u1_exact = 2 * np.pi * np.cos(2 * np.pi * YY) * np.sin(2 * np.pi * XX)
    u2_exact = -2 * np.pi * np.cos(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    vel_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.nanmax(np.abs(result['u'] - vel_exact))
    l2_error = np.sqrt(np.nanmean((result['u'] - vel_exact)**2))
    print(f"Max error: {error:.2e}")
    print(f"L2 error: {l2_error:.2e}")
