import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # --- Extract case parameters ---
    pde = case_spec["pde"]
    time_params = pde["time"]
    t0 = time_params["t0"]
    t_end = time_params["t_end"]
    dt_suggested = time_params["dt"]

    coeff = pde["coefficients"]

    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    # --- Mesh and function space ---
    mesh_res = 128
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1

    elem_degree = 2
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)

    # --- Variable kappa (time-independent) ---
    kappa_ufl = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55)**2 + (x[1] - 0.45)**2))
    kappa_func = fem.Function(V)
    kappa_func.interpolate(fem.Expression(kappa_ufl, V.element.interpolation_points))

    # --- Time constant ---
    t_const = fem.Constant(domain, ScalarType(t0))

    # --- Exact solution as UFL ---
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # --- Source term ---
    du_dt_ufl = -u_exact_ufl
    laplacian_u_ufl = -2.0 * ufl.pi**2 * u_exact_ufl
    grad_kappa_ufl = ufl.grad(kappa_ufl)
    grad_u_ufl = ufl.grad(u_exact_ufl)
    f_ufl = du_dt_ufl - kappa_ufl * laplacian_u_ufl - ufl.dot(grad_kappa_ufl, grad_u_ufl)

    # --- Functions for time-stepping ---
    f_func = fem.Function(V)
    u_bc_func = fem.Function(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)

    # --- Initial condition ---
    u_n.interpolate(fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    u_h.x.array[:] = u_n.x.array[:]

    # --- Variational form (Backward Euler) ---
    dt_val = dt_suggested
    n_steps = int(round((t_end - t0) / dt_val))

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a_ufl = (ufl.inner(u_trial, v_test) * ufl.dx
             + dt_val * kappa_func * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx)
    L_ufl = (ufl.inner(u_n, v_test) * ufl.dx
             + dt_val * ufl.inner(f_func, v_test) * ufl.dx)

    a_form = fem.form(a_ufl)
    L_form = fem.form(L_ufl)

    # --- Boundary DOFs ---
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # --- Assemble matrix ---
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # --- Solver setup ---
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
    solver.setFromOptions()

    # --- RHS vector ---
    b = petsc.create_vector(L_form.function_spaces)

    total_iterations = 0

    # --- Time loop ---
    for step in range(1, n_steps + 1):
        t_current = t0 + step * dt_val
        t_const.value = ScalarType(t_current)

        f_func.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))
        u_bc_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)

        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]

    # --- Compute L2 error ---
    t_const.value = ScalarType(t_end)
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    error_L2_sq = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx)),
        op=MPI.SUM
    )
    error_L2 = np.sqrt(error_L2_sq)

    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {error_L2:.6e}")

    # --- Sample solution on output grid ---
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

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    if comm.size > 1:
        recv_buf = np.zeros_like(u_values)
        comm.Allreduce(u_values, recv_buf, op=MPI.SUM)
        u_values = recv_buf

    u_grid = u_values.reshape(ny_out, nx_out)

    # --- Initial condition on grid ---
    u_initial_vals = np.sin(np.pi * points[0, :]) * np.sin(np.pi * points[1, :])
    u_initial_grid = u_initial_vals.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01},
            "coefficients": {
                "kappa": {"type": "expr", "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))"}
            }
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}
        }
    }
    import time as _time
    _t0 = _time.time()
    result = solve(case_spec)
    _t1 = _time.time()
    print(f"Wall time: {_t1-_t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
