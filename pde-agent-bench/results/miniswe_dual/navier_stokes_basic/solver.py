import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time as time_module

def solve(case_spec: dict) -> dict:
    t_start = time_module.time()
    
    nu_val = 0.1
    N = 48
    degree_u = 3
    degree_p = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
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
    
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact))
    
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    V_sub, _ = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    Q_sub, _ = W.sub(1).collapse()
    def origin(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), origin)
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    w.sub(0).interpolate(u_exact_expr)
    w.x.scatter_forward()
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 30,
            "snes_linesearch_type": "bt",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    
    problem.solve()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    converged_reason = snes.getConvergedReason()
    assert converged_reason > 0, f"SNES did not converge, reason: {converged_reason}"
    
    u_h = w.sub(0).collapse()
    
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    vel_mag = np.zeros(nx_eval * ny_eval)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[i] = np.sqrt(ux**2 + uy**2)
    
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
    import time as t
    case_spec = {"pde": {"type": "navier_stokes", "viscosity": 0.1}}
    start = t.time()
    result = solve(case_spec)
    elapsed = t.time() - start
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solver info: {result['solver_info']}")
    nx, ny = 50, 50
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    error = np.abs(result['u'] - vel_mag_exact)
    print(f"Max error: {error.max():.2e}")
    print(f"Mean error: {error.mean():.2e}")
    print(f"L2-like error: {np.sqrt(np.mean(error**2)):.2e}")
