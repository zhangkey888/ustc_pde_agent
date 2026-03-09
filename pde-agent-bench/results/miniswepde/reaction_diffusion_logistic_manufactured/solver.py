import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde_spec = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults
    t_end = 0.3
    dt_val = 0.01
    time_scheme = "backward_euler"
    
    time_spec = pde_spec.get("time", None)
    if time_spec is not None:
        t_end = float(time_spec.get("t_end", t_end))
        dt_val = float(time_spec.get("dt", dt_val))
        time_scheme = time_spec.get("scheme", time_scheme)

    epsilon = 1.0
    coeffs = pde_spec.get("coefficients", {})
    if isinstance(coeffs, dict):
        epsilon = float(coeffs.get("epsilon", 1.0))

    element_degree = 2
    N = 48

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi

    t_const = fem.Constant(domain, ScalarType(0.0))

    u_exact_ufl = ufl.exp(-t_const) * (0.2 + 0.1 * ufl.sin(pi_val * x[0]) * ufl.sin(pi_val * x[1]))

    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    dudt_exact = -u_exact_ufl
    R_exact = u_exact_ufl * (1.0 - u_exact_ufl)
    f_expr = dudt_exact - epsilon * laplacian_u_exact + R_exact

    u_h = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)
    
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    R_u = u_h * (1.0 - u_h)
    
    F = ((u_h - u_n) / dt_c) * v * ufl.dx \
        + eps_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_expr * v * ufl.dx

    u_bc_func = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    def update_bcs(t_val):
        t_const.value = t_val
        u_bc_func.interpolate(bc_expr)
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    bcs = [bc]

    # Initial condition
    t_const.value = 0.0
    u_n.interpolate(bc_expr)
    u_h.interpolate(bc_expr)

    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(bc_expr)

    # Setup nonlinear problem with built-in SNES
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "hypre",
    }
    
    problem = petsc.NonlinearProblem(
        F, u_h, bcs=bcs,
        petsc_options_prefix="rd_",
        petsc_options=petsc_opts,
    )

    n_steps = int(round(t_end / dt_val))
    dt_actual = t_end / n_steps
    dt_c.value = dt_actual

    nonlinear_iterations = []
    total_linear_iters = 0
    t = 0.0

    for step in range(n_steps):
        t += dt_actual
        update_bcs(t)
        
        # Solve using the built-in SNES
        problem.solve()
        u_h.x.scatter_forward()
        
        # Get iteration info from SNES
        snes = problem.solver
        n_iters = snes.getIterationNumber()
        nonlinear_iterations.append(int(n_iters))
        
        # Get linear iterations
        ksp = snes.getKSP()
        total_linear_iters += ksp.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate on output grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    u_init_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": total_linear_iters,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
            "nonlinear_iterations": nonlinear_iterations,
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "coefficients": {"epsilon": 1.0},
            "reaction": {"type": "logistic"},
            "time": {
                "t_end": 0.3,
                "dt": 0.01,
                "scheme": "backward_euler"
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    nx_out, ny_out = 60, 60
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    t_end_val = 0.3
    u_exact = np.exp(-t_end_val) * (0.2 + 0.1 * np.sin(np.pi * XX) * np.sin(np.pi * YY))
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    print(f"L2 error (RMS): {error:.6e}")
    print(f"Max error: {np.nanmax(np.abs(result['u'] - u_exact)):.6e}")
    print(f"Target error: <= 8.93e-03")
    print(f"Target time: <= 89.888s")
