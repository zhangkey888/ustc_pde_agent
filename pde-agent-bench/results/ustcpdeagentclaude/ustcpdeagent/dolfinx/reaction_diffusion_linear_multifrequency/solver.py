import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fpetsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # --- Parse case spec ---
    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {}) or case_spec.get("time", {})
    # Defaults
    t0 = 0.0
    t_end = 0.4
    dt_suggest = 0.005
    if time_info:
        t0 = float(time_info.get("t0", 0.0))
        t_end = float(time_info.get("t_end", 0.4))
        dt_suggest = float(time_info.get("dt", 0.005))

    # epsilon (diffusion coef). The manufactured solution uses some epsilon.
    # Not specified - default to 1.0 (typical).
    params = pde.get("params", {}) or case_spec.get("params", {})
    epsilon = float(params.get("epsilon", 1.0))
    # Reaction R(u) = c*u, linear. Default c=0.
    react_coef = float(params.get("reaction_coef", 0.0))

    output = case_spec["output"]["grid"]
    nx_out = output["nx"]
    ny_out = output["ny"]
    bbox = output["bbox"]

    comm = MPI.COMM_WORLD

    # --- Mesh ---
    N = 96  # increased for better accuracy
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Time step
    dt_val = 0.005  # refined for better time accuracy
    n_steps = int(round((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps

    # --- Exact solution, source term ---
    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    pi = ufl.pi

    def u_exact_ufl(t_sym):
        return ufl.exp(-t_sym) * (
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
            + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
        )

    # f = du/dt - eps*lap(u) + R(u)
    # For R(u) = c*u linear.
    u_sym = u_exact_ufl(t_const)
    # Derivative wrt time: du/dt = -u (since u = exp(-t)*G(x,y))
    dudt_sym = -u_sym
    lap_u = ufl.div(ufl.grad(u_sym))
    f_expr = dudt_sym - epsilon * lap_u + react_coef * u_sym

    # --- Trial/test ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)  # previous time
    # Initial condition: u(x,0) = sin(pi x)sin(pi y) + 0.2 sin(6pi x)sin(5pi y)
    u0_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) + \
              0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    # Source at time t_n and t_{n+1}
    t_n_const = fem.Constant(msh, PETSc.ScalarType(t0))
    t_np1_const = fem.Constant(msh, PETSc.ScalarType(t0 + dt_val))

    def f_at(t_sym):
        u_s = ufl.exp(-t_sym) * (
            ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
            + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
        )
        dudt_s = -u_s
        lap_s = ufl.div(ufl.grad(u_s))
        return dudt_s - epsilon * lap_s + react_coef * u_s

    f_n = f_at(t_n_const)
    f_np1 = f_at(t_np1_const)

    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(epsilon))
    rc = fem.Constant(msh, PETSc.ScalarType(react_coef))

    # Crank-Nicolson:
    # (u - u_n)/dt = 0.5*(eps*lap(u) - R(u) + f_np1 + eps*lap(u_n) - R(u_n) + f_n)
    # Weak form:
    # (u,v)/dt + 0.5*eps*(grad u, grad v) + 0.5*rc*(u,v)
    #   = (u_n,v)/dt - 0.5*eps*(grad u_n, grad v) - 0.5*rc*(u_n,v) + 0.5*(f_n+f_np1, v)
    a = (u * v / dt_c) * ufl.dx \
        + 0.5 * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + 0.5 * rc * u * v * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx \
        - 0.5 * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx \
        - 0.5 * rc * u_n * v * ufl.dx \
        + 0.5 * (f_n + f_np1) * v * ufl.dx

    # --- BC (Dirichlet = exact solution) ---
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    t_bc_const = fem.Constant(msh, PETSc.ScalarType(t0 + dt_val))
    u_bc_expr = ufl.exp(-t_bc_const) * (
        ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
    )
    bc_expr_compiled = fem.Expression(u_bc_expr, V.element.interpolation_points)
    u_bc.interpolate(bc_expr_compiled)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Save initial
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # --- Manual assembly for speed ---
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fpetsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fpetsc.create_vector(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)

    u_sol = fem.Function(V)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_n_const.value = t_cur
        t_np1_const.value = t_cur + dt_val
        t_bc_const.value = t_cur + dt_val
        u_bc.interpolate(bc_expr_compiled)

        with b.localForm() as loc:
            loc.set(0)
        fpetsc.assemble_vector(b, L_form)
        fpetsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fpetsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]
        t_cur += dt_val

    # --- Verify accuracy vs exact ---
    u_ex_final = ufl.exp(-fem.Constant(msh, PETSc.ScalarType(t_cur))) * (
        ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        + 0.2 * ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1])
    )
    err_form = fem.form((u_sol - u_ex_final) ** 2 * ufl.dx)
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    # --- Sample to uniform grid ---
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(XX.size)]).T

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)

    cells = []
    pts_on = []
    emap = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_on.append(pts[i])
            emap.append(i)
    vals_full = np.full(pts.shape[0], np.nan)
    if len(pts_on) > 0:
        v = u_sol.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        vals_full[emap] = v.flatten()
    u_grid = vals_full.reshape(ny_out, nx_out)

    # initial sample
    vals_init_full = np.full(pts.shape[0], np.nan)
    if len(pts_on) > 0:
        v0 = u_initial_func.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        vals_init_full[emap] = v0.flatten()
    u_init_grid = vals_init_full.reshape(ny_out, nx_out)

    info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(total_iters),
        "dt": float(dt_val),
        "n_steps": int(n_steps),
        "time_scheme": "crank_nicolson",
        "l2_error_vs_exact": float(err_l2),
        "epsilon": epsilon,
    }

    return {"u": u_grid, "solver_info": info, "u_initial": u_init_grid}
