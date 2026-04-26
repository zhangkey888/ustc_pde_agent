import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})

    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.4)
    dt_suggested = time_params.get("dt", 0.005)

    epsilon = pde.get("epsilon", 0.01)
    reaction_coeff = pde.get("reaction_coeff", 1.0)

    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    xmin, xmax, ymin, ymax = bbox

    # ---- Mesh and function space ----
    mesh_res = 48
    elem_degree = 2
    dt = dt_suggested
    n_steps = int(round((t_end - t0) / dt))

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))

    # ---- UFL expressions for exact solution and source ----
    sin = ufl.sin
    pi = ufl.pi
    exp = ufl.exp

    def u_exact_ufl(t_sym):
        return exp(-t_sym) * (sin(pi*x[0])*sin(pi*x[1]) + 0.2*sin(6*pi*x[0])*sin(5*pi*x[1]))

    def f_source_ufl(t_sym):
        u_ex = u_exact_ufl(t_sym)
        du_dt = -exp(-t_sym) * (sin(pi*x[0])*sin(pi*x[1]) + 0.2*sin(6*pi*x[0])*sin(5*pi*x[1]))
        lap_u = exp(-t_sym) * (-2*pi**2*sin(pi*x[0])*sin(pi*x[1])
                               - 0.2*61*pi**2*sin(6*pi*x[0])*sin(5*pi*x[1]))
        return du_dt - epsilon*lap_u + reaction_coeff*u_ex

    def u_exact_numpy(t, X):
        return np.exp(-t) * (np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
                             + 0.2*np.sin(6*np.pi*X[0])*np.sin(5*np.pi*X[1]))

    # ---- Precompile Expressions (JIT once) ----
    ip = V.element.interpolation_points
    u_exact_expr = fem.Expression(u_exact_ufl(t_const), ip)
    f_source_expr = fem.Expression(f_source_ufl(t_const), ip)

    # ---- Variational forms (compiled once) ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # LHS: (u,v)/dt + 0.5*eps*(grad(u),grad(v)) + 0.5*R*(u,v)
    a_form = (ufl.inner(u, v)/dt
              + 0.5*epsilon*ufl.inner(ufl.grad(u), ufl.grad(v))
              + 0.5*reaction_coeff*ufl.inner(u, v)) * ufl.dx
    a_compiled = fem.form(a_form)

    # RHS uses u_n and f_func (fem.Function updated each step)
    u_n = fem.Function(V)
    f_func = fem.Function(V)
    L_form = (ufl.inner(u_n, v)/dt
              - 0.5*epsilon*ufl.inner(ufl.grad(u_n), ufl.grad(v))
              - 0.5*reaction_coeff*ufl.inner(u_n, v)
              + ufl.inner(f_func, v)) * ufl.dx
    L_compiled = fem.form(L_form)

    # ---- Dirichlet BC on entire boundary ----
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    g_bc = fem.Function(V)
    t_const.value = ScalarType(t0)
    g_bc.interpolate(u_exact_expr)
    bc = fem.dirichletbc(g_bc, boundary_dofs)

    # ---- Assemble LHS matrix (constant, with BC) ----
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # ---- KSP solver ----
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.SOR)
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    # ---- Initialize u_n ----
    t_const.value = ScalarType(t0)
    u_n.interpolate(u_exact_expr)

    # ---- Prepare output point evaluation (once) ----
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    pts_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)

    # ---- Time stepping ----
    b = petsc.create_vector(L_compiled.function_spaces)
    u_h = fem.Function(V)
    total_ksp_iterations = 0
    t = t0

    for step in range(n_steps):
        t_next = t + dt
        t_mid = t + 0.5 * dt

        # Update f at midpoint
        t_const.value = ScalarType(t_mid)
        f_func.interpolate(f_source_expr)

        # Assemble RHS
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set BC
        t_const.value = ScalarType(t_next)
        g_bc.interpolate(u_exact_expr)
        petsc.set_bc(b, [bc])

        # Solve
        ksp.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_ksp_iterations += ksp.getIterationNumber()

        # Update
        u_n.x.array[:] = u_h.x.array[:]
        t = t_next

    # ---- Sample final solution on output grid ----
    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()

    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global

    u_grid = u_grid.reshape(ny_out, nx_out)

    # ---- Compute L2 error for verification ----
    t_const.value = ScalarType(t_end)
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(u_exact_expr)
    error_L2 = domain.comm.allreduce(
        np.fabs(fem.assemble_scalar(fem.form((u_h - u_ex_func)**2 * ufl.dx)))**0.5,
        op=MPI.SUM
    )
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {error_L2:.6e}")

    # ---- Sample initial condition ----
    t_const.value = ScalarType(t0)
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)

    u_init_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_grid[eval_map] = vals_init.flatten()
    if comm.size > 1:
        u_init_global = np.zeros_like(u_init_grid)
        comm.Allreduce(u_init_grid, u_init_global, op=MPI.SUM)
        u_init_grid = u_init_global
    u_init_grid = u_init_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "sor",
        "rtol": 1e-10,
        "iterations": total_ksp_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson",
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }
