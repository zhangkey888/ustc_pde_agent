import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa = float(pde["coefficients"]["kappa"])

    time_params = pde["time"]
    t0 = float(time_params["t0"])
    t_end = float(time_params["t_end"])
    dt_suggested = float(time_params["dt"])

    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 200
    deg = 2
    dt = dt_suggested / 2.0
    n_steps = int(round((t_end - t0) / dt))

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))

    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    t_var = fem.Constant(domain, PETSc.ScalarType(t0))

    u_exact = ufl.exp(-t_var) * ufl.exp(-40.0 * r2)
    f_val = u_exact * (160.0 * kappa - 1.0 - 6400.0 * kappa * r2)
    g_val = ufl.exp(-t_var) * ufl.exp(-40.0 * r2)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(ufl.exp(-40.0 * r2), V.element.interpolation_points))

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a = u_trial * v_test * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = (u_n + dt * f_val) * v_test * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    fdim = domain.topology.dim - 1
    bnd_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bnd_dofs = fem.locate_dofs_topological(V, fdim, bnd_facets)

    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(g_val, V.element.interpolation_points))
    bc = fem.dirichletbc(g_func, bnd_dofs)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    rtol_val = 1e-10
    ksp.setTolerances(rtol=rtol_val, atol=1e-12)

    u_sol = fem.Function(V)
    b_vec = petsc.create_vector(L_form.function_spaces)

    total_iters = 0

    for step in range(n_steps):
        t_current = t0 + (step + 1) * dt
        t_var.value = PETSc.ScalarType(t_current)

        g_func.interpolate(fem.Expression(g_val, V.element.interpolation_points))

        with b_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[[bc]])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, [bc])

        ksp.solve(b_vec, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iters += ksp.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()

    bb = geometry.bb_tree(domain, domain.topology.dim)
    cands = geometry.compute_collisions_points(bb, pts.T)
    cells = geometry.compute_colliding_cells(domain, cands, pts.T)

    p_list = []
    c_list = []
    idx_map = []
    for i in range(pts.shape[1]):
        lk = cells.links(i)
        if len(lk) > 0:
            p_list.append(pts.T[i])
            c_list.append(lk[0])
            idx_map.append(i)

    u_grid_flat = np.zeros(pts.shape[1])
    if len(p_list) > 0:
        ev = u_sol.eval(np.array(p_list), np.array(c_list, dtype=np.int32))
        u_grid_flat[idx_map] = ev.flatten()

    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid_flat)
        comm.Allreduce(u_grid_flat, u_grid_global, op=MPI.SUM)
        u_grid_flat = u_grid_global

    u_grid = u_grid_flat.reshape(ny_out, nx_out)

    # Also sample initial condition
    u_init_expr = ufl.exp(-40.0 * r2)
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))

    u_init_flat = np.zeros(pts.shape[1])
    if len(p_list) > 0:
        ev_init = u_init_func.eval(np.array(p_list), np.array(c_list, dtype=np.int32))
        u_init_flat[idx_map] = ev_init.flatten()

    if comm.size > 1:
        u_init_global = np.zeros_like(u_init_flat)
        comm.Allreduce(u_init_flat, u_init_global, op=MPI.SUM)
        u_init_flat = u_init_global

    u_initial = u_init_flat.reshape(ny_out, nx_out)

    # Compute L2 error against exact solution
    t_var.value = PETSc.ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    err_diff = fem.Function(V)
    err_diff.x.array[:] = u_sol.x.array[:] - u_exact_func.x.array[:]
    L2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_diff, err_diff) * ufl.dx))
    L2_sq = comm.allreduce(L2_sq, op=MPI.SUM)
    L2_error = np.sqrt(L2_sq)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol_val,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "L2_error": L2_error,
        }
    }
