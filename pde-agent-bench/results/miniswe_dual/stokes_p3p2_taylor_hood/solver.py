import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc


def solve(case_spec: dict = None) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec
    nu_val = 1.0
    nx_eval = 100
    ny_eval = 100
    degree_u = 3
    degree_p = 2

    if case_spec is not None:
        nu_val = case_spec.get('pde', {}).get('viscosity', 1.0)
        output = case_spec.get('output', {})
        nx_eval = output.get('nx', 100)
        ny_eval = output.get('ny', 100)

    # With P3/P2 elements, a moderate mesh should give very high accuracy
    N = 20  # Start small, P3 is high order

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Taylor-Hood P3/P2 mixed element
    cell_name = domain.topology.cell_name()
    V_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(2,))
    Q_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    mel = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, mel)

    # Also create separate spaces for BC interpolation
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    pi_s = ufl.pi
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Manufactured exact solution (UFL)
    u_exact_0 = pi_s * ufl.cos(pi_s * x[1]) * ufl.sin(pi_s * x[0])
    u_exact_1 = -pi_s * ufl.cos(pi_s * x[0]) * ufl.sin(pi_s * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.cos(pi_s * x[0]) * ufl.cos(pi_s * x[1])

    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f_expr = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Bilinear form: Stokes
    a_form = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              - p * ufl.div(v) * ufl.dx
              - q * ufl.div(u) * ufl.dx)

    L_form = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = u_exact on all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    bcs = [bc_u]

    # Solve with direct solver (most robust for saddle-point)
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    # Extract velocity
    uh = wh.sub(0).collapse()

    # Evaluate on grid
    xv = np.linspace(0, 1, nx_eval)
    yv = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xv, yv, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_vals = np.full((points_3d.shape[0], domain.geometry.dim), np.nan)

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
        for idx, mi in enumerate(eval_map):
            u_vals[mi, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        },
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min/max: {np.nanmin(result['u']):.6f} / {np.nanmax(result['u']):.6f}")
    print(f"solver_info: {result['solver_info']}")

    # Compute error against exact solution
    nx, ny = 100, 100
    xv = np.linspace(0, 1, nx)
    yv = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xv, yv, indexing='ij')

    u_exact_0 = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u_exact_1 = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u_exact_0**2 + u_exact_1**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2)) / np.sqrt(np.mean(vel_mag_exact**2))
    print(f"Relative L2 error: {error:.2e}")
    
    max_error = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    print(f"Max absolute error: {max_error:.2e}")
