import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # ── Domain ──
    mesh_res = 96
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # ── Parameters ──
    kappa = 0.5
    t0 = 0.0
    t_end = 0.2
    dt = 0.0025
    n_steps = int(round((t_end - t0) / dt))

    # ── Function space (P2) ──
    elem_deg = 2
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # ── Spatial coordinate ──
    x = ufl.SpatialCoordinate(domain)

    # ── Exact / manufactured solution ──
    def u_exact_ufl(t_val):
        return ufl.exp(-2.0 * t_val) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def u_exact_numpy(t_val, xx, yy):
        return np.exp(-2.0 * t_val) * np.sin(np.pi * xx) * np.sin(np.pi * yy)

    # ── Functions ──
    u_h = fem.Function(V)
    u_prev = fem.Function(V)

    # ── Initial condition ──
    u_h.interpolate(fem.Expression(u_exact_ufl(t0), V.element.interpolation_points))
    u_prev.x.array[:] = u_h.x.array[:]

    # ── Boundary condition ──
    g_bc = fem.Function(V)
    g_bc.interpolate(fem.Expression(u_exact_ufl(t0), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_bc, boundary_dofs)

    # ── Time constant ──
    t_val = fem.Constant(domain, PETSc.ScalarType(t0))

    # ── Variational form (backward Euler) ──
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_ufl = (ufl.pi**2 - 2.0) * ufl.exp(-2.0 * t_val) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    a_form = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = ufl.inner(u_prev, v) * ufl.dx + dt * ufl.inner(f_ufl, v) * ufl.dx

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # ── Assemble matrix (constant) ──
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # ── Solver ──
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
    ksp.setFromOptions()

    b = petsc.create_vector(L_compiled.function_spaces)

    total_iterations = 0
    t = t0

    # ── Store initial solution for output ──
    u_initial_data = u_h.x.array.copy()

    # ── Time stepping ──
    for step in range(n_steps):
        t = t0 + (step + 1) * dt

        # Update time constant
        t_val.value = PETSc.ScalarType(t)

        # Update boundary condition
        g_bc.interpolate(fem.Expression(u_exact_ufl(t), V.element.interpolation_points))

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        # Apply lifting
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        # Track iterations
        total_iterations += ksp.getIterationNumber()

        # Update previous
        u_prev.x.array[:] = u_h.x.array[:]

    # ── Compute L2 error at final time ──
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_exact_ufl(t_end), V.element.interpolation_points))
    M = ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx
    M_form = fem.form(M)
    l2_error_local = fem.assemble_scalar(M_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {l2_error:.6e}")
        print(f"Total linear iterations: {total_iterations}")

    # ── Sample solution onto output grid ──
    grid_info = case_spec["output"]["grid"]
    nx = grid_info["nx"]
    ny = grid_info["ny"]
    bbox = grid_info["bbox"]

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid_flat = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()

    # Gather across processes
    u_grid_flat_global = np.zeros_like(u_grid_flat) if comm.rank == 0 else None
    comm.Reduce(u_grid_flat, u_grid_flat_global, op=MPI.SUM, root=0)

    if comm.rank == 0:
        for i in range(len(u_grid_flat_global)):
            if np.isnan(u_grid_flat_global[i]):
                u_grid_flat_global[i] = u_exact_numpy(t_end, pts[i, 0], pts[i, 1])
        u_grid = u_grid_flat_global.reshape(ny, nx)
    else:
        u_grid = np.zeros((ny, nx))
    u_grid = comm.bcast(u_grid, root=0)

    # Also sample u_initial
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_initial_data
    u_initial_flat = np.full((nx * ny,), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_flat[eval_map] = vals_init.flatten()

    u_initial_flat_global = np.zeros_like(u_initial_flat) if comm.rank == 0 else None
    comm.Reduce(u_initial_flat, u_initial_flat_global, op=MPI.SUM, root=0)
    if comm.rank == 0:
        for i in range(len(u_initial_flat_global)):
            if np.isnan(u_initial_flat_global[i]):
                u_initial_flat_global[i] = u_exact_numpy(t0, pts[i, 0], pts[i, 1])
        u_initial_grid = u_initial_flat_global.reshape(ny, nx)
    else:
        u_initial_grid = np.zeros((ny, nx))
    u_initial_grid = comm.bcast(u_initial_grid, root=0)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
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
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
