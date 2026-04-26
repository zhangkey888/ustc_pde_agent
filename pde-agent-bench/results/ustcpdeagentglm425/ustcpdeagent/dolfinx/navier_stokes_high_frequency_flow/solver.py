import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

comm = MPI.COMM_WORLD

def solve(case_spec: dict) -> dict:
    # ── Extract case parameters ──
    pde = case_spec["pde"]
    nu = float(pde.get("viscosity", pde.get("nu", 0.1)))

    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # ── Mesh ──
    n_cells = 256
    msh = mesh.create_unit_square(comm, n_cells, n_cells, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # ── Mixed function space (Taylor-Hood P2/P1) ──
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # ── Exact solution and source term ──
    x_c = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_ex = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x_c[1])*ufl.sin(2*pi*x_c[0]),
        -2*pi*ufl.cos(2*pi*x_c[0])*ufl.sin(2*pi*x_c[1])
    ])
    p_ex = ufl.sin(2*pi*x_c[0])*ufl.cos(2*pi*x_c[1])

    def eps(u):
        return ufl.sym(ufl.grad(u))

    f_source = -2*nu*ufl.div(eps(u_ex)) + ufl.grad(u_ex)*u_ex + ufl.grad(p_ex)

    # ── Boundary conditions ──
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)

    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bc_p_list = []
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bc_p_list.append(bc_p)

    bcs = [bc_u] + bc_p_list

    # ── Step 1: Stokes solve as initial guess ──
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v_s, q_s) = ufl.TestFunctions(W)

    a_stokes = (2*nu*ufl.inner(eps(u_s), eps(v_s))*ufl.dx
                - ufl.inner(p_s, ufl.div(v_s))*ufl.dx
                + ufl.inner(ufl.div(u_s), q_s)*ufl.dx)
    L_stokes = ufl.inner(f_source, v_s)*ufl.dx

    stokes_problem = fem_petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    w_stokes_h = stokes_problem.solve()
    w_stokes_h.x.scatter_forward()

    # ── Step 2: Newton solve ──
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (2*nu*ufl.inner(eps(u), eps(v))*ufl.dx
         + ufl.inner(ufl.grad(u)*u, v)*ufl.dx
         - ufl.inner(p, ufl.div(v))*ufl.dx
         + ufl.inner(ufl.div(u), q)*ufl.dx
         - ufl.inner(f_source, v)*ufl.dx)

    J = ufl.derivative(F, w)

    # Initialize with Stokes solution
    w.x.array[:] = w_stokes_h.x.array[:]
    w.x.scatter_forward()

    # Try Newton solve
    newton_its = 0
    linear_its_total = 0
    newton_converged = False

    try:
        petsc_options_ns = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 20,
            "snes_converged_reason": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_converged_reason": None,
        }

        problem_ns = fem_petsc.NonlinearProblem(
            F, w, bcs=bcs, J=J,
            petsc_options_prefix="ns_",
            petsc_options=petsc_options_ns
        )

        w_h = problem_ns.solve()
        w.x.scatter_forward()

        snes = problem_ns._snes
        newton_its = int(snes.getIterationNumber())
        linear_its_total = int(snes.getLinearSolveIterations())
        newton_converged = True

        if comm.rank == 0:
            print(f"Newton converged in {newton_its} iterations, {linear_its_total} linear its")

    except Exception as e:
        if comm.rank == 0:
            print(f"Newton failed: {e}, falling back to Picard")

    # ── Fallback: Picard iteration ──
    if not newton_converged:
        w_k = fem.Function(W)
        w_k.x.array[:] = w_stokes_h.x.array[:]
        w_k.x.scatter_forward()

        u_k_sub = fem.Function(V)
        u_k_sub.interpolate(w_k.sub(0))

        (u_new, p_new) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        a_picard = (2*nu*ufl.inner(eps(u_new), eps(v))*ufl.dx
                    + ufl.inner(ufl.grad(u_new)*u_k_sub, v)*ufl.dx
                    - ufl.inner(p_new, ufl.div(v))*ufl.dx
                    + ufl.inner(ufl.div(u_new), q)*ufl.dx)
        L_picard = ufl.inner(f_source, v)*ufl.dx

        picard_problem = fem_petsc.LinearProblem(
            a_picard, L_picard, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix="picard_"
        )

        picard_max_it = 50
        picard_tol = 1e-8
        picard_its = 0

        for it in range(picard_max_it):
            w_new = picard_problem.solve()
            w_new.x.scatter_forward()

            diff = w_new.x.array - w_k.x.array
            rel_err = np.linalg.norm(diff) / max(np.linalg.norm(w_k.x.array), 1e-14)

            picard_its += 1

            if comm.rank == 0:
                print(f"Picard iter {it}: rel_err = {rel_err:.6e}")

            if rel_err < picard_tol:
                break

            w_k.x.array[:] = w_new.x.array[:]
            w_k.x.scatter_forward()
            u_k_sub.interpolate(w_k.sub(0))

        w.x.array[:] = w_k.x.array[:]
        w.x.scatter_forward()
        newton_its = picard_its

    # ── Compute L2 error for verification ──
    V_err, _ = W.sub(0).collapse()
    Q_err, _ = W.sub(1).collapse()

    u_err_func = fem.Function(V_err)
    u_err_expr = fem.Expression(u_ex, V_err.element.interpolation_points)
    u_err_func.interpolate(u_err_expr)

    u_computed = fem.Function(V_err)
    u_computed.interpolate(w.sub(0))

    err_u_sq = fem.assemble_scalar(fem.form(ufl.inner(u_computed - u_err_func, u_computed - u_err_func)*ufl.dx))
    err_u_sq = msh.comm.allreduce(err_u_sq, op=MPI.SUM)

    u_ex_sq = fem.assemble_scalar(fem.form(ufl.inner(u_err_func, u_err_func)*ufl.dx))
    u_ex_sq = msh.comm.allreduce(u_ex_sq, op=MPI.SUM)

    l2_err_u = np.sqrt(err_u_sq)
    rel_err_u = np.sqrt(err_u_sq / u_ex_sq) if u_ex_sq > 0 else l2_err_u

    if comm.rank == 0:
        print(f"Velocity L2 error: {l2_err_u:.6e}, Relative: {rel_err_u:.6e}")

    # ── Sample solution on output grid ──
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.zeros((nx_out * ny_out, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_sol = fem.Function(V_err)
    u_sol.interpolate(w.sub(0))

    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_eval = np.array(points_on_proc)
        cells_eval = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_eval, cells_eval)
        u_vals[eval_map] = vals

    u_vals_global = np.zeros_like(u_vals)
    comm.Allreduce(u_vals, u_vals_global, op=MPI.SUM)
    u_vals_global = np.nan_to_num(u_vals_global, nan=0.0)

    u_magnitude = np.sqrt(u_vals_global[:, 0]**2 + u_vals_global[:, 1]**2)
    u_grid = u_magnitude.reshape(ny_out, nx_out)

    # ── Solver info ──
    solver_info = {
        "mesh_resolution": n_cells,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": linear_its_total if newton_converged else 0,
        "nonlinear_iterations": [newton_its],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
