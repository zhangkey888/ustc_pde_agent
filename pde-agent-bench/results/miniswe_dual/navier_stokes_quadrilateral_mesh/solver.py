import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = 0.1
    N = 32
    degree_u = 3
    degree_p = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    
    cell_type = domain.basix_cell()
    vel_el = basix.ufl.element("Lagrange", cell_type, degree_u, shape=(2,))
    pres_el = basix.ufl.element("Lagrange", cell_type, degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact))
    
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V_col, V_map = W.sub(0).collapse()
    
    u_bc_func = fem.Function(V_col)
    u_exact_expr = fem.Expression(u_exact, V_col.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    Q_col, Q_map = W.sub(1).collapse()
    
    def origin_marker(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_col), origin_marker)
    p_bc_func = fem.Function(Q_col)
    p_bc_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate exact solution for velocity
    w_init_u = fem.Function(V_col)
    w_init_u.interpolate(u_exact_expr)
    w.x.array[V_map] = w_init_u.x.array
    w.x.scatter_forward()
    
    # Solve using new NonlinearProblem API with MUMPS
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    
    problem.solve()
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    reason = snes.getConvergedReason()
    assert reason > 0, f"SNES did not converge, reason: {reason}"
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_magnitude[idx]
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve({})
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")
    print(f"u_grid shape: {result['u'].shape}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
    print(f"Solver info: {result['solver_info']}")
    
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"RMS error: {error:.2e}")
    print(f"Max error: {max_error:.2e}")
