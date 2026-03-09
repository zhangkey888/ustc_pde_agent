import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nu_val = 1.0
    N = 64
    degree_u = 2
    degree_p = 1

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Create mixed Taylor-Hood element
    vel_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pres_elem = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mixed_el = basix.ufl.mixed_element([vel_elem, pres_elem])

    W = fem.functionspace(domain, mixed_el)

    # Collapsed sub-spaces for BCs and post-processing
    V_sub, V_map = W.sub(0).collapse()
    Q_sub, Q_map = W.sub(1).collapse()

    # Test and trial functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)

    # Exact solution (for source term and BCs)
    u_exact_expr = ufl.as_vector([
        ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1]),
        -ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact_expr = ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])

    # Source term: f = -nu * lap(u_exact) + grad(p_exact)
    f_expr = -nu_val * ufl.div(ufl.grad(u_exact_expr)) + ufl.grad(p_exact_expr)

    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Bilinear form
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)

    # Linear form
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: exact velocity on all boundaries
    u_bc_func = fem.Function(V_sub)
    u_bc_expr_interp = fem.Expression(u_exact_expr, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr_interp)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc]

    # Solve using LinearProblem
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

    # Extract velocity sub-function
    u_sol = wh.sub(0).collapse()

    # Evaluate on 100x100 grid
    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points_2d = np.zeros((3, nx_eval * ny_eval))
    points_2d[0, :] = XX.flatten()
    points_2d[1, :] = YY.flatten()

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    u_values = np.full((nx_eval * ny_eval, domain.geometry.dim), np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            u_values[i, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    vel_mag_grid = vel_mag.reshape((nx_eval, ny_eval))

    return {
        "u": vel_mag_grid,
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
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")

    # Compute error against exact solution
    nx_eval, ny_eval = 100, 100
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    u1_exact = np.pi * np.exp(XX) * np.cos(np.pi * YY)
    u2_exact = -np.exp(XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2)) / np.sqrt(np.mean(vel_mag_exact**2))
    print(f"Relative L2 error: {error:.2e}")

    abs_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"Max absolute error: {abs_error:.2e}")
