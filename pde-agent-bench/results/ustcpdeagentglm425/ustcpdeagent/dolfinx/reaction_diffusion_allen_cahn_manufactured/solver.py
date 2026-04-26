import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as wall_time

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    start_time = wall_time.time()

    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})

    # Time parameters
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.15)
    dt_suggested = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "backward_euler")

    # Output grid parameters
    nx_out = grid_spec.get("nx", 50)
    ny_out = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Agent-selectable parameters with optimized defaults
    epsilon = case_spec.get("epsilon", 0.01)
    mesh_res = case_spec.get("mesh_resolution", 60)
    element_degree = case_spec.get("element_degree", 2)
    newton_rtol = case_spec.get("newton_rtol", 1e-10)
    dt = case_spec.get("dt", 0.005)

    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps  # exact dt

    # Create mesh and function space
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    # Solution and test functions
    u = fem.Function(V)
    u_n = fem.Function(V)
    v = ufl.TestFunction(V)

    # Time constant for UFL expressions
    t_const = fem.Constant(domain, ScalarType(t0))

    # Manufactured solution UFL expression
    u_exact_ufl = 0.3 * ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term f derived from manufactured solution
    # PDE: du/dt - eps*laplacian(u) + (u^3 - u) = f
    # f = u_exact^3 - 2*u_exact + 2*eps*pi^2*u_exact
    f_source = u_exact_ufl**3 - 2*u_exact_ufl + 2*epsilon*pi**2*u_exact_ufl

    # Residual: backward Euler
    F = (u - u_n) / dt * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u**3 - u) * v * ufl.dx \
        - f_source * v * ufl.dx

    # Jacobian
    J = ufl.derivative(F, u)

    # Boundary conditions - Dirichlet on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Pre-compile ALL expressions before time loop to avoid JIT in loop
    t_const.value = ScalarType(t0)
    ic_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    final_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # Set initial condition
    u_n.interpolate(ic_expr)
    u.x.array[:] = u_n.x.array[:]

    # Save initial condition for output
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]

    # Setup nonlinear solver with LU for robustness
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="ac_",
        petsc_options=petsc_options
    )

    # Time stepping
    nonlinear_iters = []
    total_linear_iters = 0

    for step in range(n_steps):
        current_t = t0 + (step + 1) * dt
        t_const.value = ScalarType(current_t)

        # Update BC to exact solution at current time
        u_bc.interpolate(bc_expr)

        # Solve nonlinear system
        u_sol = problem.solve()
        u.x.scatter_forward()

        # Track iterations
        snes = problem._snes
        newton_iters = snes.getIterationNumber()
        linear_iters = snes.getLinearSolveIterations()
        nonlinear_iters.append(int(newton_iters))
        total_linear_iters += linear_iters

        # Update previous solution
        u_n.x.array[:] = u.x.array[:]

    # Compute L2 error for verification
    t_const.value = ScalarType(t_end)
    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(final_expr)

    diff_expr = (u - u_exact_final)**2 * ufl.dx
    error_form = fem.form(diff_expr)
    error_local = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    elapsed = wall_time.time() - start_time

    # Sample solution on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    # Point evaluation using bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Sample initial condition on same grid
    u_init_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()

    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    # Handle parallel: gather results from all processes
    if comm.size > 1:
        all_values = comm.allgather(u_values)
        gathered = np.full((points.shape[0],), np.nan)
        for proc_vals in all_values:
            mask = ~np.isnan(proc_vals)
            gathered[mask] = proc_vals[mask]
        u_grid = gathered.reshape(ny_out, nx_out)

        all_init = comm.allgather(u_init_values)
        gathered_init = np.full((points.shape[0],), np.nan)
        for proc_vals in all_init:
            mask = ~np.isnan(proc_vals)
            gathered_init[mask] = proc_vals[mask]
        u_initial_grid = gathered_init.reshape(ny_out, nx_out)

    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": float(newton_rtol),
        "iterations": total_linear_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iters,
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
