import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module


def solve(case_spec: dict) -> dict:
    wall_start = time_module.time()

    # ================================================================
    # Extract parameters from case_spec
    # ================================================================
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})

    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox

    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.3))
    dt_suggested = float(time_params.get("dt", 0.005))

    epsilon = float(pde.get("epsilon", 0.01))
    kappa = float(pde.get("kappa", 1.0))

    # ================================================================
    # Adaptive mesh resolution, element degree, and time step
    # based on epsilon (boundary layer strength)
    # ================================================================
    if epsilon >= 0.05:
        mesh_res = 64
        elem_degree = 2
        dt_factor = 2
    elif epsilon >= 0.01:
        mesh_res = 96
        elem_degree = 2
        dt_factor = 4
    elif epsilon >= 0.005:
        mesh_res = 128
        elem_degree = 2
        dt_factor = 6
    elif epsilon >= 0.001:
        mesh_res = 192
        elem_degree = 2
        dt_factor = 8
    else:
        mesh_res = 256
        elem_degree = 2
        dt_factor = 10

    dt_use = dt_suggested / dt_factor
    n_steps = max(1, int(round((t_end - t0) / dt_use)))
    dt_actual = (t_end - t0) / n_steps

    # ================================================================
    # Create mesh
    # ================================================================
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle
    )
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)

    # ================================================================
    # Function space
    # ================================================================
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    u_sol = fem.Function(V)
    u_prev = fem.Function(V)

    # ================================================================
    # Spatial coordinate
    # ================================================================
    x = ufl.SpatialCoordinate(domain)

    # ================================================================
    # Source term f (spatially varying):
    # u = exp(-t) * exp(4x) * sin(pi*y)
    # f = (-1 - eps*(16-pi^2) + kappa) * exp(-t)*exp(4x)*sin(pi*y)
    # ================================================================
    coeff_f = -1.0 - epsilon * (16.0 - np.pi**2) + kappa

    # ================================================================
    # Time variable as a Constant
    # ================================================================
    t_var = fem.Constant(domain, PETSc.ScalarType(t0))

    # ================================================================
    # Pre-compile expressions
    # ================================================================
    g_expr_ufl = ufl.exp(-t_var) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    g_expression = fem.Expression(g_expr_ufl, V.element.interpolation_points)

    u_init_ufl = ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    u_init_expression = fem.Expression(u_init_ufl, V.element.interpolation_points)

    u_exact_ufl = ufl.exp(-t_end) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact_expression = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # ================================================================
    # Dirichlet BC on entire boundary
    # ================================================================
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    g_func = fem.Function(V)
    g_func.interpolate(g_expression)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    # ================================================================
    # Initial condition
    # ================================================================
    u_prev.interpolate(u_init_expression)
    u_sol.x.array[:] = u_prev.x.array[:]

    # ================================================================
    # Weak form: Backward Euler
    # (u - u_prev)/dt * v + eps * grad(u) . grad(v) + kappa * u * v = f * v
    # f = coeff_f * exp(-t_var) * exp(4x) * sin(pi*y)
    # ================================================================
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_form = (1.0 / dt_actual) * ufl.inner(u_trial, v_test) * ufl.dx \
             + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx \
             + kappa * ufl.inner(u_trial, v_test) * ufl.dx

    f_ufl = coeff_f * ufl.exp(-t_var) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    L_form = (1.0 / dt_actual) * ufl.inner(u_prev, v_test) * ufl.dx \
             + ufl.inner(f_ufl, v_test) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix (constant for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # Create solver with LU
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()

    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1)

    # Create RHS vector
    b = petsc.create_vector(L_compiled.function_spaces)

    total_iterations = 0

    # ================================================================
    # Time stepping loop
    # ================================================================
    for step in range(n_steps):
        t_current = t0 + (step + 1) * dt_actual

        # Update time constant
        t_var.value = PETSc.ScalarType(t_current)

        # Update BC function
        g_func.interpolate(g_expression)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        # Apply lifting
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update u_prev
        u_prev.x.array[:] = u_sol.x.array[:]

    # ================================================================
    # Compute L2 error for verification
    # ================================================================
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_expression)

    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_sq = comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM)
    l2_error = np.sqrt(l2_error_sq) if l2_error_sq > 0 else 0.0

    # ================================================================
    # Sample solution onto output grid
    # ================================================================
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

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

    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()

    u_grid_global = np.zeros_like(u_grid)
    comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
    u_grid_global = np.nan_to_num(u_grid_global, nan=0.0)
    u_grid_2d = u_grid_global.reshape(ny_out, nx_out)

    # Sample initial condition onto output grid
    u_init_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        u_init_vals = np.exp(4.0 * pts_arr[:, 0]) * np.sin(np.pi * pts_arr[:, 1])
        u_init_grid[eval_map] = u_init_vals

    u_init_global = np.zeros_like(u_init_grid)
    comm.Allreduce(u_init_grid, u_init_global, op=MPI.SUM)
    u_init_global = np.nan_to_num(u_init_global, nan=0.0)
    u_init_2d = u_init_global.reshape(ny_out, nx_out)

    wall_end = time_module.time()
    wall_time = wall_end - wall_start

    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"Wall time: {wall_time:.2f}s")
        print(f"Mesh res: {mesh_res}, elem_degree: {elem_degree}, dt: {dt_actual:.6f}, n_steps: {n_steps}")
        print(f"Total iterations: {total_iterations}")

    # ================================================================
    # Return results
    # ================================================================
    result = {
        "u": u_grid_2d,
        "u_initial": u_init_2d,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }

    return result
