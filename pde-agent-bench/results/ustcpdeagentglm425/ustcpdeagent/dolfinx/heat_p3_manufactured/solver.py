import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # --- Parse case spec ---
    pde = case_spec["pde"]
    kappa_val = pde["coefficients"]["kappa"]
    t0 = pde["time"]["t0"]
    t_end = pde["time"]["t_end"]
    dt_suggested = pde["time"]["dt"]
    scheme = pde["time"]["scheme"]  # backward_euler

    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # --- Choose discretisation parameters ---
    mesh_res = 80
    element_degree = 3
    dt = dt_suggested / 2.0  # halve dt for better accuracy
    n_steps = int(round((t_end - t0) / dt))

    # --- Create mesh ---
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # --- Function space ---
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # --- Exact / manufactured solution helpers ---
    x_spatial = ufl.SpatialCoordinate(domain)
    pi = np.pi

    def u_exact_expr(t):
        return ufl.exp(-t) * ufl.sin(pi * x_spatial[0]) * ufl.sin(2 * pi * x_spatial[1])

    def f_source_expr(t):
        # f = du/dt - div(kappa * grad(u))
        # = -exp(-t)*sin(pi*x)*sin(2*pi*y) - kappa*(-pi^2 - 4*pi^2)*exp(-t)*sin(pi*x)*sin(2*pi*y)
        # = (-1 + 5*pi^2*kappa) * exp(-t)*sin(pi*x)*sin(2*pi*y)
        return ScalarType(-1.0 + 5.0 * pi**2 * kappa_val) * ufl.exp(-t) * ufl.sin(pi * x_spatial[0]) * ufl.sin(2 * pi * x_spatial[1])

    # --- Boundary condition (Dirichlet on entire boundary) ---
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # --- Time-stepping setup (Backward Euler) ---
    u = fem.Function(V)   # solution at n+1
    u_n = fem.Function(V) # solution at n

    # Interpolate initial condition
    u_n.interpolate(fem.Expression(u_exact_expr(t0), V.element.interpolation_points))
    u_n.x.scatter_forward()

    # Store initial condition for output
    u_initial_vals = u_n.x.array.copy()

    # Variational form
    v = ufl.TestFunction(V)
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    dt_const = fem.Constant(domain, ScalarType(dt))

    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f  =>  (u,v)/dt + kappa*(grad(u),grad(v)) = (u_n,v)/dt + (f,v)
    a_form = (ufl.inner(u, v) / dt_const + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L_form = (ufl.inner(u_n, v) / dt_const + ufl.inner(f_source_expr(ufl.Constant(domain, ScalarType(0.0))), v)) * ufl.dx

    # We need a time constant that we update each step
    t_const = fem.Constant(domain, ScalarType(0.0))

    # Re-define L with the time constant properly
    f_at_t = ScalarType(-1.0 + 5.0 * pi**2 * kappa_val) * ufl.exp(t_const) * ufl.sin(pi * x_spatial[0]) * ufl.sin(2 * pi * x_spatial[1])
    # Note: exp(-t) = exp(t_const) when t_const = -t... let me be more careful
    # f(t) = (-1 + 5*pi^2*kappa) * exp(-t) * sin(pi*x)*sin(2*pi*y)
    # Let me use t_const = -t so exp(t_const) = exp(-t)
    # Actually let me just define it directly

    f_ufl = ScalarType(-1.0 + 5.0 * pi**2 * kappa_val) * ufl.exp(-t_const) * ufl.sin(pi * x_spatial[0]) * ufl.sin(2 * pi * x_spatial[1])

    L_form = (ufl.inner(u_n, v) / dt_const + ufl.inner(f_ufl, v)) * ufl.dx
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix (time-independent)
    A = petsc.assemble_matrix(a_compiled, bcs=[])
    A.assemble()

    # Solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14)
    solver.setMaxIt(500)

    b = petsc.create_vector(L_compiled.function_spaces)

    total_iterations = 0
    t_current = t0

    for step in range(n_steps):
        t_next = t_current + dt
        t_const.value = ScalarType(t_next)

        # Update BC
        g_bc = fem.Function(V)
        g_bc.interpolate(fem.Expression(u_exact_expr(t_next), V.element.interpolation_points))
        bc = fem.dirichletbc(g_bc, boundary_dofs)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        # Apply lifting
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        # Count iterations
        total_iterations += solver.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u.x.array[:]
        u_n.x.scatter_forward()
        t_current = t_next

    # --- Compute L2 error at t_end ---
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_expr(t_end), V.element.interpolation_points))
    u_exact_func.x.scatter_forward()

    error_form = fem.form(ufl.inner(u - u_exact_func, u - u_exact_func) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {error_l2:.6e}")

    # --- Sample solution on output grid ---
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_flat = np.zeros((nx_out * ny_out, 3), dtype=np.float64)
    pts_flat[:, 0] = XX.ravel()
    pts_flat[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_flat)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts_flat.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_flat[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u.eval(pts_arr, cells_arr)
        u_grid[idx_map] = vals.flatten()

    # Gather across ranks
    if comm.size > 1:
        from mpi4py import MPI as MPI4PY
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI4PY.SUM)
        # Handle NaN: if a point is not on any proc, sum of NaNs is NaN
        # Use a mask approach
        nan_mask = np.isnan(u_grid)
        u_grid_int = np.where(nan_mask, 0.0, u_grid)
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid_int, u_grid_global, op=MPI4PY.SUM)
        u_grid = u_grid_global

    u_grid = u_grid.reshape(ny_out, nx_out)

    # --- Sample initial condition on output grid ---
    u_initial_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        u_n_for_ic = fem.Function(V)
        u_n_for_ic.x.array[:] = u_initial_vals
        vals_ic = u_n_for_ic.eval(pts_arr, cells_arr)
        u_initial_grid[idx_map] = vals_ic.flatten()

    if comm.size > 1:
        nan_mask_ic = np.isnan(u_initial_grid)
        u_initial_int = np.where(nan_mask_ic, 0.0, u_initial_grid)
        u_initial_global = np.zeros_like(u_initial_grid)
        comm.Allreduce(u_initial_int, u_initial_global, op=MPI4PY.SUM)
        u_initial_grid = u_initial_global

    u_initial_grid = u_initial_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }
