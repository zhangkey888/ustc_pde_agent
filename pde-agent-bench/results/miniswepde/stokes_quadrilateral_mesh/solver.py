import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nu_val = 1.0
    N = 32
    degree_u = 3
    degree_p = 2

    # Create quadrilateral mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Create mixed function space (Taylor-Hood: Q3/Q2)
    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    el_p = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mel = basix.ufl.mixed_element([el_u, el_p])
    W = fem.functionspace(domain, mel)

    # Collapsed sub-spaces for BCs and interpolation
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    # Define test and trial functions
    (v, q) = ufl.TestFunctions(W)
    (u, p) = ufl.TrialFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Manufactured solution
    u_exact_expr = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact_expr = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    # Compute source term from manufactured solution
    f = -nu_val * ufl.div(ufl.grad(u_exact_expr)) + ufl.grad(p_exact_expr)

    # Bilinear form
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions: u = u_exact on all boundaries
    def all_boundary(x_coord):
        return (np.isclose(x_coord[0], 0.0) | np.isclose(x_coord[0], 1.0) |
                np.isclose(x_coord[1], 0.0) | np.isclose(x_coord[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, all_boundary)

    # Interpolate exact velocity
    u_bc_func = fem.Function(V)
    u_exact_interp = fem.Expression(u_exact_expr, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_interp)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at corner (0,0) to fix the constant
    p_dofs_all = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x_coord: np.isclose(x_coord[0], 0.0) & np.isclose(x_coord[1], 0.0)
    )
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(
        lambda x_coord: np.full(x_coord.shape[1],
                                np.cos(np.pi * x_coord[0]) * np.cos(np.pi * x_coord[1]))
    )
    bc_p = fem.dirichletbc(p_bc_func, p_dofs_all, W.sub(1))

    bcs = [bc_u, bc_p]

    # Solve with direct solver
    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()

    # Extract velocity
    uh = wh.sub(0).collapse()

    # Evaluate on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])

    # dolfinx needs 3D points
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    # Evaluate velocity at points
    u_values = np.full((points_3d.shape[0], domain.geometry.dim), np.nan)
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
        vals = uh.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            u_values[i, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    # Compute error against exact solution
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"RMS error: {error:.2e}")
    print(f"Max error: {max_error:.2e}")
