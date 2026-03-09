import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu_val = 0.1
    if 'pde' in case_spec:
        pde = case_spec['pde']
        if 'viscosity' in pde:
            nu_val = float(pde['viscosity'])

    N = 64
    degree_u = 3
    degree_p = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    P_u_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    P_p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    ME = basix.ufl.mixed_element([P_u_el, P_p_el])
    W = fem.functionspace(domain, ME)

    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    x = ufl.SpatialCoordinate(domain)
    pi_ufl = ufl.pi

    u1_exact = pi_ufl * ufl.cos(pi_ufl * x[1]) * ufl.sin(pi_ufl * x[0]) + pi_ufl * ufl.cos(4*pi_ufl * x[1]) * ufl.sin(2*pi_ufl * x[0])
    u2_exact = -pi_ufl * ufl.cos(pi_ufl * x[0]) * ufl.sin(pi_ufl * x[1]) - (pi_ufl/2) * ufl.cos(2*pi_ufl * x[0]) * ufl.sin(4*pi_ufl * x[1])
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    p_exact = ufl.sin(pi_ufl * x[0]) * ufl.cos(2*pi_ufl * x[1])

    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    f = (ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact))

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F_form = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
              - p * ufl.div(v) * ufl.dx
              + q * ufl.div(u) * ufl.dx
              - ufl.inner(f, v) * ufl.dx)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    p_bc_func = fem.Function(Q)
    p_exact_expr_q = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr_q)

    def origin_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), origin_marker)
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))

    bcs = [bc_u, bc_p]

    V_collapsed, dofs_V = W.sub(0).collapse()
    Q_collapsed, dofs_Q = W.sub(1).collapse()

    u_init = fem.Function(V_collapsed)
    u_init_expr = fem.Expression(u_exact, V_collapsed.element.interpolation_points)
    u_init.interpolate(u_init_expr)

    p_init = fem.Function(Q_collapsed)
    p_init_expr = fem.Expression(p_exact, Q_collapsed.element.interpolation_points)
    p_init.interpolate(p_init_expr)

    w.x.array[dofs_V] = u_init.x.array
    w.x.array[dofs_Q] = p_init.x.array
    w.x.scatter_forward()

    problem = petsc.NonlinearProblem(
        F_form, w,
        bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )

    problem.solve()
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    w.x.scatter_forward()

    u_sol = w.sub(0).collapse()

    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx_out * ny_out, domain.geometry.dim), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    vel_mag_grid = vel_mag.reshape((nx_out, ny_out))

    return {
        "u": vel_mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [int(n_newton)],
        }
    }


if __name__ == "__main__":
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Newton iterations: {result['solver_info']['nonlinear_iterations']}")

    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    pi = np.pi
    u1_ex = pi * np.cos(pi * YY) * np.sin(pi * XX) + pi * np.cos(4*pi * YY) * np.sin(2*pi * XX)
    u2_ex = -pi * np.cos(pi * XX) * np.sin(pi * YY) - (pi/2) * np.cos(2*pi * XX) * np.sin(4*pi * YY)
    vel_mag_exact = np.sqrt(u1_ex**2 + u2_ex**2)

    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2)) / np.sqrt(np.mean(vel_mag_exact**2))
    print(f"Relative L2 error: {error:.2e}")
    abs_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"Max absolute error: {abs_error:.2e}")
