import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the reaction-diffusion (Allen-Cahn) equation."""
    
    start_time = time.time()
    
    # Parse case spec
    pde_spec = case_spec.get("pde", {})
    
    # Time parameters - hardcoded defaults from problem description
    time_spec = pde_spec.get("time", {})
    t_end = time_spec.get("t_end", 0.2)
    dt_val = time_spec.get("dt", 0.005)
    time_scheme = time_spec.get("scheme", "backward_euler")
    is_transient = True
    
    # Diffusion coefficient
    epsilon = pde_spec.get("coefficients", {}).get("epsilon", 1.0)
    
    # Mesh and element parameters
    N = 64
    degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Time variable as a Constant
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    
    # Manufactured solution: u = exp(-t)*(0.25*sin(2*pi*x)*sin(pi*y))
    u_exact_ufl = ufl.exp(-t) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # du/dt of exact solution
    du_dt_exact = -ufl.exp(-t) * (0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]))
    
    # R(u) for Allen-Cahn: u^3 - u
    R_exact = u_exact_ufl**3 - u_exact_ufl
    
    # Source term: f = du/dt - epsilon * div(grad(u_exact)) + R(u_exact)
    f_expr = du_dt_exact - epsilon * ufl.div(ufl.grad(u_exact_ufl)) + R_exact
    
    # Define solution functions
    u_n = fem.Function(V, name="u_n")
    u_h = fem.Function(V, name="u_h")
    v = ufl.TestFunction(V)
    
    # Initial condition: u(x, 0) = 0.25 * sin(2*pi*x) * sin(pi*y)
    u_init_expr = fem.Expression(
        0.25 * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)
    u_h.interpolate(u_init_expr)
    
    # Backward Euler nonlinear residual
    R_u = u_h**3 - u_h
    
    F = ((u_h - u_n) / dt_c) * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx \
        + R_u * v * ufl.dx \
        - f_expr * v * ufl.dx
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    
    t.value = 0.0
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Time stepping
    n_steps = int(round(t_end / dt_val))
    current_t = 0.0
    nonlinear_iterations = []
    total_linear_iters = 0
    
    # We'll create a new NonlinearProblem each time step to update BCs properly,
    # or we can reuse one and just update the BC function values
    
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "snes_linesearch_type": "bt",
    }
    
    # Create the nonlinear problem once
    problem = petsc.NonlinearProblem(
        F, u_h, bcs=bcs,
        petsc_options_prefix="allen_cahn_",
        petsc_options=petsc_opts,
    )
    
    for step in range(n_steps):
        current_t += dt_val
        t.value = current_t
        
        # Update boundary condition
        u_bc.interpolate(bc_expr)
        
        # Solve
        problem.solve()
        snes = problem.solver
        reason = snes.getConvergedReason()
        n_iters = snes.getIterationNumber()
        
        if reason <= 0:
            print(f"Warning: SNES did not converge at step {step}, reason={reason}")
        
        u_h.x.scatter_forward()
        nonlinear_iterations.append(n_iters)
        
        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]
    
    # Evaluate on 60x60 grid
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
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
    
    # Evaluate initial condition on same grid
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_init_expr)
    
    u_init_values = np.full(points_3d.shape[0], np.nan)
    pts_on2 = []
    cells_on2 = []
    emap2 = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_on2.append(points_3d[i])
            cells_on2.append(links[0])
            emap2.append(i)
    if len(pts_on2) > 0:
        vals2 = u_init_func.eval(np.array(pts_on2), np.array(cells_on2, dtype=np.int32))
        u_init_values[emap2] = vals2.flatten()
    
    result = {
        "u": u_grid,
        "u_initial": u_init_values.reshape((nx_out, ny_out)),
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_linear_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }
    
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "reaction_diffusion",
            "reaction": {"type": "allen_cahn"},
            "coefficients": {"epsilon": 1.0},
            "time": {
                "t_end": 0.2,
                "dt": 0.005,
                "scheme": "backward_euler"
            },
            "manufactured_solution": "exp(-t)*(0.25*sin(2*pi*x)*sin(pi*y))"
        },
        "domain": {
            "type": "unit_square",
            "dim": 2
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    
    nx_out, ny_out = 60, 60
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    t_end_val = 0.2
    u_exact = np.exp(-t_end_val) * 0.25 * np.sin(2 * np.pi * XX) * np.sin(np.pi * YY)
    
    error = np.sqrt(np.nanmean((result['u'] - u_exact)**2))
    max_error = np.nanmax(np.abs(result['u'] - u_exact))
    print(f"RMS error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"Newton iterations per step: {result['solver_info']['nonlinear_iterations']}")
