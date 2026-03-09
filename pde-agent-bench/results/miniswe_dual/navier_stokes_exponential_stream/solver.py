import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    """Solve steady incompressible Navier-Stokes with manufactured solution."""
    
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 0.15
    
    N = 40
    degree_u = 3
    degree_p = 2
    
    t_start = time.time()
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Create mixed function space (Taylor-Hood P3/P2)
    cell_name = domain.topology.cell_name()
    Pu_vec = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(domain.geometry.dim,))
    Pp_el = basix.ufl.element("Lagrange", cell_name, degree_p)
    ME = basix.ufl.mixed_element([Pu_vec, Pp_el])
    W = fem.functionspace(domain, ME)
    
    # Individual space for BCs
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    
    # Define unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution (for source term computation via UFL)
    pi_ufl = ufl.pi
    u_exact = ufl.as_vector([
        pi_ufl * ufl.exp(2 * x[0]) * ufl.cos(pi_ufl * x[1]),
        -2.0 * ufl.exp(2 * x[0]) * ufl.sin(pi_ufl * x[1])
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(pi_ufl * x[1])
    
    # Compute source term symbolically: f = (u_exact · ∇)u_exact - ν Δu_exact + ∇p_exact
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.dot(grad_u_exact, u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)
    f = convection_exact - nu_val * laplacian_u_exact + grad_p_exact
    
    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Weak form (residual)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.exp(2 * x[0]) * np.cos(np.pi * x[1]),
        -2.0 * np.exp(2 * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess: zero
    w.x.array[:] = 0.0
    w.x.scatter_forward()
    
    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w,
        bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    problem.solve()
    reason = problem.solver.getConvergedReason()
    newton_its = problem.solver.getIterationNumber()
    assert reason > 0, f"Newton solver did not converge: reason={reason}"
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    vel_mag = np.full(points_3d.shape[1], np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        vel_mag_local = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_mag_local[idx]
    
    u_grid = vel_mag.reshape((nx_eval, ny_eval))
    
    elapsed = time.time() - t_start
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "nonlinear_iterations": [newton_its],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    t0 = time.time()
    result = solve({})
    total = time.time() - t0
    print(f"Output shape: {result['u'].shape}")
    print(f"Min/Max velocity magnitude: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
    print(f"Total time: {total:.3f}s")
    print(f"Solver info: {result['solver_info']}")
    
    # Verify against exact solution at grid points
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u1_exact = np.pi * np.exp(2 * XX) * np.cos(np.pi * YY)
    u2_exact = -2.0 * np.exp(2 * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    max_err = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    rms_err = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    print(f"Max pointwise error in vel_mag: {max_err:.6e}")
    print(f"RMS error in vel_mag: {rms_err:.6e}")
