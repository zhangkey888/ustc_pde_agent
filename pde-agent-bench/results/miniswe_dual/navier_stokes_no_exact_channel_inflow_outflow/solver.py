import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
import basix
from petsc4py import PETSc


def solve(case_spec: dict = None):
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    nu_val = float(pde_spec.get("viscosity", 0.12))
    
    output_spec = case_spec.get("output", {})
    nx_out = int(output_spec.get("nx", 50))
    ny_out = int(output_spec.get("ny", 50))
    
    # Adaptive mesh refinement loop
    resolutions = [48, 80, 128]
    degree_u = 2
    degree_p = 1
    
    prev_grid = None
    final_result = None
    final_info = None
    
    for N in resolutions:
        result, info = _solve_at_resolution(N, degree_u, degree_p, nu_val, nx_out, ny_out)
        
        if prev_grid is not None:
            # Check convergence
            valid = ~np.isnan(result) & ~np.isnan(prev_grid)
            if np.any(valid):
                max_diff = np.max(np.abs(result[valid] - prev_grid[valid]))
                if max_diff < 1e-5:
                    return {"u": result, "solver_info": info}
        
        prev_grid = result.copy()
        final_result = result
        final_info = info
    
    return {"u": final_result, "solver_info": final_info}


def _solve_at_resolution(N, degree_u, degree_p, nu_val, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed space
    V_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(2,))
    Q_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    
    # --- Boundary Conditions ---
    # Inflow: parabolic on left (x=0)
    inflow_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    dofs_inflow = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
    bc_inflow = fem.dirichletbc(u_inflow, dofs_inflow, W.sub(0))
    
    # No-slip on walls (y=0, y=1)
    wall_facets = mesh.locate_entities_boundary(domain, fdim,
        lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
    u_noslip = fem.Function(V)
    u_noslip.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
    dofs_wall = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    bc_wall = fem.dirichletbc(u_noslip, dofs_wall, W.sub(0))
    
    # Outflow: zero pressure on right (x=1) - "do-nothing" / stress-free
    outflow_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    p_out = fem.Function(Q)
    p_out.interpolate(lambda x: np.zeros_like(x[0]))
    dofs_p_out = fem.locate_dofs_topological((W.sub(1), Q), fdim, outflow_facets)
    bc_p_out = fem.dirichletbc(p_out, dofs_p_out, W.sub(1))
    
    bcs = [bc_inflow, bc_wall, bc_p_out]
    
    # Weak form: steady Navier-Stokes
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    # Initial guess: parabolic profile (close to exact for Poiseuille flow)
    w.sub(0).interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    
    # Newton solver with direct LU
    prefix = f'ns{N}_'
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix=prefix,
        petsc_options={
            'snes_type': 'newtonls',
            'snes_rtol': 1e-10,
            'snes_atol': 1e-12,
            'snes_max_it': 50,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
    )
    
    problem.solve()
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    converged_reason = snes.getConvergedReason()
    
    if converged_reason < 0:
        # Retry with relaxation and line search
        w.x.array[:] = 0.0
        w.sub(0).interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
        prefix2 = f'ns{N}r_'
        problem2 = petsc.NonlinearProblem(
            F_form, w, bcs=bcs,
            petsc_options_prefix=prefix2,
            petsc_options={
                'snes_type': 'newtonls',
                'snes_rtol': 1e-8,
                'snes_atol': 1e-10,
                'snes_max_it': 100,
                'snes_linesearch_type': 'bt',
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
            }
        )
        problem2.solve()
        n_newton = problem2.solver.getIterationNumber()
    
    w.x.scatter_forward()
    u_sol = w.sub(0).collapse()
    
    # Evaluate on output grid
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [int(n_newton)],
    }
    
    return u_grid, solver_info


def _evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate velocity magnitude on a uniform nx x ny grid over [0,1]^2."""
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xx.flatten()
    points[1, :] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx * ny, 2), np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        vals = vals.reshape(-1, 2)
        for idx, map_idx in enumerate(eval_map):
            u_values[map_idx, :] = vals[idx, :]
    
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    return vel_mag.reshape(nx, ny)


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    info = result["solver_info"]
    print(f"Time: {elapsed:.2f}s")
    print(f"Grid shape: {u_grid.shape}")
    print(f"Min: {np.nanmin(u_grid):.8f}, Max: {np.nanmax(u_grid):.8f}, Mean: {np.nanmean(u_grid):.8f}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
    print(f"Info: {info}")
