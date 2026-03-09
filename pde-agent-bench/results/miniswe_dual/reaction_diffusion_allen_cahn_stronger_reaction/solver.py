import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

def solve(case_spec: dict) -> dict:
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.002))
    time_scheme = time_spec.get("scheme", "backward_euler")
    params = pde_spec.get("parameters", {})
    epsilon = float(params.get("epsilon", 0.01))
    reaction_lambda = float(params.get("reaction_lambda", 5.0))
    agent_params = case_spec.get("agent_params", {})
    N = int(agent_params.get("mesh_resolution", 64))
    degree = int(agent_params.get("element_degree", 2))
    dt = float(agent_params.get("dt", dt_suggested))

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    t_const = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Manufactured solution
    u_exact_ufl = ufl.exp(-t_const) * (0.15 + 0.12 * ufl.sin(2*pi*x[0]) * ufl.sin(2*pi*x[1]))
    dudt = -u_exact_ufl
    lap_u_exact = ufl.div(ufl.grad(u_exact_ufl))
    f_expr = dudt - epsilon * lap_u_exact + reaction_lambda * (u_exact_ufl**3 - u_exact_ufl)

    # BC and IC
    u_bc_func = fem.Function(V)
    u_exact_interp = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n = fem.Function(V)
    t_const.value = 0.0
    u_n.interpolate(u_exact_interp)
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    # Weak form - Backward Euler
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    F = (u_h - u_n) / dt_const * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + reaction_lambda * (u_h**3 - u_h) * v * ufl.dx \
        - f_expr * v * ufl.dx

    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]

    # Time stepping
    t = 0.0
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const.value = actual_dt
    nonlinear_iterations = []

    # We need to create the NonlinearProblem once, but update BCs each step
    # Since the problem references u_h, u_n, t_const which we update in-place, 
    # we can reuse the problem object
    
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "ksp_rtol": 1e-8,
        "snes_linesearch_type": "bt",
    }

    problem = petsc.NonlinearProblem(F, u_h, petsc_options_prefix="ac_", bcs=bcs, petsc_options=petsc_opts)

    for step in range(n_steps):
        t += actual_dt
        t_const.value = t
        u_bc_func.interpolate(u_exact_interp)
        u_h.x.array[:] = u_n.x.array[:]
        problem.solve()
        snes = problem.solver
        reason = snes.getConvergedReason()
        n_iters = snes.getIterationNumber()
        nonlinear_iterations.append(n_iters)
        u_h.x.scatter_forward()
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate on output grid
    nx_out, ny_out = 75, 75
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_values = np.full(points_3d.shape[0], np.nan)
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
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))

    # Initial condition on grid
    t_const.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_interp)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals2.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N, "element_degree": degree,
            "ksp_type": "gmres", "pc_type": "hypre", "rtol": 1e-8,
            "iterations": 0, "dt": actual_dt, "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    return result

if __name__ == "__main__":
    case_spec = {"pde": {"time": {"t_end": 0.1, "dt": 0.002, "scheme": "backward_euler"}, "parameters": {"epsilon": 0.01, "reaction_lambda": 5.0}}}
    t0 = time_module.time()
    result = solve(case_spec)
    elapsed = time_module.time() - t0
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    nx, ny = 75, 75
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * (0.15 + 0.12 * np.sin(2*np.pi*XX) * np.sin(2*np.pi*YY))
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {np.nanmax(np.abs(result['u'] - u_exact)):.6e}")
    print(f"Newton iters per step: {result['solver_info']['nonlinear_iterations'][:5]}...")
