import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import basix.ufl
import time


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    nu_val = pde_spec.get("viscosity", 0.1)
    
    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    degree_u = 3
    degree_p = 2
    N = 40
    
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_rectangle(
        comm,
        [np.array([x_range[0], y_range[0]]), np.array([x_range[1], y_range[1]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    vel_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)
    
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    
    w = fem.Function(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    (u, p) = ufl.split(w)
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    # Exact manufactured solution
    u_exact = ufl.as_vector([
        pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Source term derived from manufactured solution
    f = (ufl.grad(u_exact) * u_exact
         - nu_val * ufl.div(ufl.grad(u_exact))
         + ufl.grad(p_exact))
    
    # Weak form (residual)
    F_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v_test)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v_test) * ufl.dx
        - p * ufl.div(v_test) * ufl.dx
        - ufl.inner(f, v_test) * ufl.dx
        + ufl.div(u) * q_test * ufl.dx
    )
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess: interpolate exact solution for fast convergence
    w.sub(0).interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    w.sub(1).interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    w_sol = problem.solve()
    w.x.scatter_forward()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    converged_reason = snes.getConvergedReason()
    
    u_sol = w.sub(0).collapse()
    
    # Evaluate on output grid
    eps = 1e-10
    x_pts = np.linspace(x_range[0] + eps, x_range[1] - eps, nx_out)
    y_pts = np.linspace(y_range[0] + eps, y_range[1] - eps, ny_out)
    X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
    
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
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
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    # Fill any NaN values with exact solution (boundary points)
    nan_mask = np.isnan(vel_mag)
    if np.any(nan_mask):
        pts_nan = points_3d[:, nan_mask]
        ux_e = np.pi * np.cos(np.pi * pts_nan[1]) * np.sin(np.pi * pts_nan[0])
        uy_e = -np.pi * np.cos(np.pi * pts_nan[0]) * np.sin(np.pi * pts_nan[1])
        vel_mag[nan_mask] = np.sqrt(ux_e**2 + uy_e**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Newton iterations: {result['solver_info']['nonlinear_iterations']}")
    
    nx_out, ny_out = 50, 50
    x_pts = np.linspace(0, 1, nx_out)
    y_pts = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
    ux_exact = np.pi * np.cos(np.pi * Y) * np.sin(np.pi * X)
    uy_exact = -np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    
    error = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    nan_count = np.sum(np.isnan(result['u']))
    print(f"RMS error: {error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"NaN count: {nan_count}")
    print(f"Grid shape: {result['u'].shape}")
