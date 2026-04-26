import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec["pde"]
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]

    time_params = pde.get("time", {})
    is_transient = time_params is not None and bool(time_params)
    t0 = float(time_params.get("t0", 0.0)) if is_transient else 0.0
    t_end = float(time_params.get("t_end", 0.3)) if is_transient else 0.0
    dt_suggested = float(time_params.get("dt", 0.005)) if is_transient else 0.01

    epsilon = float(pde.get("epsilon", 1.0))
    reaction_coeff = float(pde.get("reaction_coeff", 1.0))

    mesh_res = 48
    element_degree = 2
    dt_actual = dt_suggested

    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(msh)

    f_coeff = -1.0 + 25.0 * np.pi**2 * epsilon + reaction_coeff

    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    t_mid_const = fem.Constant(msh, PETSc.ScalarType(t0))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1])
    f_mid_ufl = f_coeff * ufl.exp(-t_mid_const) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1])

    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    f_mid_expr = fem.Expression(f_mid_ufl, V.element.interpolation_points)
    u_init_expr = fem.Expression(ufl.sin(4*ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1]), V.element.interpolation_points)

    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    u_bc_func = fem.Function(V)
    f_mid_func = fem.Function(V)

    u_n.interpolate(u_init_expr)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    t_const.value = t0
    u_bc_func.interpolate(u_exact_expr)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_lhs = (ufl.inner(u_trial, v_test) / dt_actual) * ufl.dx \
          + 0.5 * epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx \
          + 0.5 * reaction_coeff * ufl.inner(u_trial, v_test) * ufl.dx

    L_rhs = (ufl.inner(u_n, v_test) / dt_actual) * ufl.dx \
          - 0.5 * epsilon * ufl.inner(ufl.grad(u_n), ufl.grad(v_test)) * ufl.dx \
          - 0.5 * reaction_coeff * ufl.inner(u_n, v_test) * ufl.dx \
          + ufl.inner(f_mid_func, v_test) * ufl.dx

    a_form = fem.form(a_lhs)
    L_form = fem.form(L_rhs)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-10

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14)

    b = petsc.create_vector(L_form.function_spaces)

    n_steps = int(round((t_end - t0) / dt_actual))
    total_iterations = 0

    for step in range(n_steps):
        t_current = t0 + (step + 1) * dt_actual
        t_mid_val = t0 + step * dt_actual + dt_actual / 2.0

        t_const.value = t_current
        t_mid_const.value = t_mid_val

        u_bc_func.interpolate(u_exact_expr)
        f_mid_func.interpolate(f_mid_expr)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]

    # L2 error
    t_const.value = t_end
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_expr)
    error_L2 = np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)), op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    if comm.size > 1:
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=MPI.SUM)
        u_values = u_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition
    u_init_vals = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        u_n_init = fem.Function(V)
        u_n_init.interpolate(u_init_expr)
        vals_init = u_n_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vals_init.flatten()
    if comm.size > 1:
        u_init_global = np.zeros_like(u_init_vals)
        comm.Allreduce(u_init_vals, u_init_global, op=MPI.SUM)
        u_init_vals = u_init_global
    u_initial_grid = u_init_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson"
        },
        "u_initial": u_initial_grid
    }
