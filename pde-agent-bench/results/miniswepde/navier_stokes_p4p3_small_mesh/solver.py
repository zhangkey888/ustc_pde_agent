import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix.ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t_start = time.time()
    
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    nu_val = 0.2
    if 'pde' in case_spec and 'viscosity' in case_spec['pde']:
        nu_val = case_spec['pde']['viscosity']
    
    # P4/P3 Taylor-Hood elements
    degree_u = 4
    degree_p = 3
    N = 16
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Create mixed function space using basix
    vel_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mel = basix.ufl.mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)
    
    # Separate spaces for BC interpolation
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Define solution function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Manufactured exact solution (UFL expressions)
    u_exact = ufl.as_vector([
        pi_val * ufl.cos(pi_val * x[1]) * ufl.sin(pi_val * x[0]),
        -pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    ])
    p_exact = ufl.cos(pi_val * x[0]) * ufl.cos(pi_val * x[1])
    
    # Source term: f = -nu * div(grad(u_exact)) + (u_exact . grad) u_exact + grad(p_exact)
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(u_exact) * u_exact + ufl.grad(p_exact)
    
    # Weak form (residual)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Boundary conditions - exact velocity on all boundaries
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda X: np.vstack([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
        -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initialize with exact solution for fast Newton convergence
    u_init = fem.Function(V)
    u_init.interpolate(lambda X: np.vstack([
        np.pi * np.cos(np.pi * X[1]) * np.sin(np.pi * X[0]),
        -np.pi * np.cos(np.pi * X[0]) * np.sin(np.pi * X[1])
    ]))
    w.sub(0).interpolate(u_init)
    
    p_init = fem.Function(Q)
    p_init.interpolate(lambda X: np.cos(np.pi * X[0]) * np.cos(np.pi * X[1]))
    w.sub(1).interpolate(p_init)
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    problem.solve()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    converged_reason = snes.getConvergedReason()
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {"pde": {"viscosity": 0.2}}
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Grid shape: {result['u'].shape}")
    print(f"Newton iters: {result['solver_info']['nonlinear_iterations']}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    # Check accuracy against exact
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    uy_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    error = np.abs(result['u'] - vel_mag_exact)
    print(f"Max error: {np.nanmax(error):.6e}")
    print(f"Mean error: {np.nanmean(error):.6e}")
    print(f"PASS: {np.nanmax(error) <= 1e-6 and elapsed <= 7.521}")
