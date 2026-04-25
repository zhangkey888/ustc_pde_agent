import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    time_params = pde["time"]
    grid_spec = case_spec["output"]["grid"]

    t0 = float(time_params["t0"])
    t_end = float(time_params["t_end"])
    dt_suggested = float(time_params["dt"])

    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    # Parameters - moderate accuracy improvement over baseline
    mesh_res = 120
    elem_deg = 2
    dt = dt_suggested / 2.0
    n_steps = int(round((t_end - t0) / dt))
    if n_steps < 1:
        n_steps = 1
        dt = t_end - t0

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    x = ufl.SpatialCoordinate(domain)

    # Variable kappa
    kappa_expr = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
    kappa_func = fem.Function(V)
    kappa_func.interpolate(fem.Expression(kappa_expr, V.element.interpolation_points))

    t_const = fem.Constant(domain, ScalarType(t0))

    # Manufactured solution and source
    u_exact = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    du_dt = -ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    f_expr = du_dt - ufl.div(kappa_expr * ufl.grad(u_exact))
    g_expr = u_exact
    u0_expr = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])

    # Bilinear and linear forms
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_ufl = (1.0/dt) * ufl.inner(u_trial, v) * ufl.dx + ufl.inner(kappa_func * ufl.grad(u_trial), ufl.grad(v)) * ufl.dx

    f_func = fem.Function(V)
    u_prev = fem.Function(V)
    L_ufl = ufl.inner(f_func, v) * ufl.dx + (1.0/dt) * ufl.inner(u_prev, v) * ufl.dx

    # BCs
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # Initial condition
    u_prev.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))

    u_sol = fem.Function(V)

    # Compile and assemble
    a_compiled = fem.form(a_ufl)
    L_compiled = fem.form(L_ufl)

    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol, atol=1e-12)
    ksp.setFromOptions()

    b = A.createVecRight()
    total_iterations = 0

    for step in range(n_steps):
        current_t = t0 + (step + 1) * dt
        t_const.value = ScalarType(current_t)

        u_bc_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
        f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()

        u_prev.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    u_values = np.full((nx_out * ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_values = domain.comm.allreduce(u_values, op=MPI.SUM)
    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition on grid
    u_init_values = np.full((nx_out * ny_out,), np.nan)
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_init_values = domain.comm.allreduce(u_init_values, op=MPI.SUM)
    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
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
