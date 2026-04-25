import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as petsc_fem
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    kappa_val = float(pde["coefficients"]["kappa"])
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])

    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]

    mesh_res = 155
    element_degree = 2
    dt = 0.001

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    u_n = fem.Function(V)
    u_sol = fem.Function(V)

    u_n.interpolate(
        fem.Expression(
            ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
            V.element.interpolation_points
        )
    )

    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    inv_dt = fem.Constant(domain, PETSc.ScalarType(1.0/dt))

    a = inv_dt * ufl.inner(u_trial, v_test) * ufl.dx + kappa_val * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    a_form = fem.form(a)

    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    f_expr = ufl.exp(-t_const) * (65*ufl.pi**2 - 1) * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])

    L = inv_dt * ufl.inner(u_n, v_test) * ufl.dx + ufl.inner(f_expr, v_test) * ufl.dx
    L_form = fem.form(L)

    g_func = fem.Function(V)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    A = petsc_fem.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.rtol = 1e-10

    b = petsc_fem.create_vector(L_form.function_spaces)

    t = t0
    n_steps = 0
    total_iterations = 0

    while t < t_end - 1e-12:
        t += dt
        n_steps += 1

        t_const.value = PETSc.ScalarType(t)

        g_func.interpolate(
            fem.Expression(
                ufl.exp(-t_const) * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
                V.element.interpolation_points
            )
        )

        with b.localForm() as loc:
            loc.set(0)
        petsc_fem.assemble_vector(b, L_form)
        petsc_fem.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc_fem.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]

    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx_out * ny_out,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Sample initial condition
    u_init_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals2 = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals2.flatten()

    u_initial_grid = u_init_values.reshape(ny_out, nx_out)

    # L2 error verification
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(8*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    error_form = fem.form(ufl.inner(u_sol - u_exact_expr, u_sol - u_exact_expr) * ufl.dx)
    error_L2 = fem.assemble_scalar(error_form)
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2, op=MPI.SUM))

    print(f"L2 error at t={t:.4f}: {error_L2:.6e}")
    print(f"Time steps: {n_steps}, Linear iterations: {total_iterations}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
