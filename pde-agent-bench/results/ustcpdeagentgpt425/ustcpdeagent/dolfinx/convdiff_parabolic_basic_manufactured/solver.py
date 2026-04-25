import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.1))
    dt_user = float(pde.get("dt", 0.02))
    eps = float(pde.get("epsilon", 0.1))
    beta_in = pde.get("beta", [1.0, 0.5])
    beta_vec = np.array(beta_in, dtype=np.float64)

    mesh_resolution = 96
    element_degree = 1
    dt = min(dt_user, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(eps))
    beta_c = fem.Constant(domain, np.asarray(beta_vec, dtype=ScalarType))

    pi = np.pi
    sx = ufl.sin(pi * x[0])
    sy = ufl.sin(pi * x[1])
    cx = ufl.cos(pi * x[0])
    cy = ufl.cos(pi * x[1])
    exp_t = ufl.exp(-t_c)

    f_expr = exp_t * (
        (-1.0 + 2.0 * eps * pi * pi) * sx * sy
        + pi * beta_vec[0] * cx * sy
        + pi * beta_vec[1] * sx * cy
    )

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-t0) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)

    def interpolate_exact_at_time(func, tval: float):
        func.interpolate(lambda X: np.exp(-tval) * np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    interpolate_exact_at_time(u_bc, t0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta_norm = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    h = ufl.CellDiameter(domain)
    if beta_norm > 1.0e-14:
        tau = 1.0 / ufl.sqrt(
            (2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (36.0 * eps_c / h**2) ** 2
        )
    else:
        tau = ScalarType(0.0)

    stream_v = ufl.dot(beta_c, ufl.grad(v))
    Lu = (u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    rhs_hist = (u_n / dt_c) + f_expr

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * stream_v * ufl.dx
    )
    L = rhs_hist * v * ufl.dx + tau * rhs_hist * stream_v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)

    total_iterations = 0
    start = time.perf_counter()

    u_initial_grid = _sample_to_grid(u_n, domain, nx_out, ny_out, bbox)

    for step in range(1, n_steps + 1):
        tnow = t0 + step * dt
        t_c.value = ScalarType(tnow)
        interpolate_exact_at_time(u_bc, tnow)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            ksp.solve(b, uh.x.petsc_vec)
            if ksp.getConvergedReason() <= 0:
                raise RuntimeError("iterative solve failed")
        except Exception:
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(ksp.getIterationNumber())
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start

    u_ex_fun = fem.Function(V)
    interpolate_exact_at_time(u_ex_fun, t_end)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_fun.x.array
    err_fun.x.scatter_forward()
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    u_grid = _sample_to_grid(uh, domain, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": 1e-8,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_err),
        "wall_time_sec": float(wall),
    }

    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


def _sample_to_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_ids, dtype=np.int32)] = vals

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = gathered[0].copy()
        for arr in gathered[1:]:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged[np.isnan(merged)] = 0.0
        out = merged.reshape(ny, nx)
    else:
        out = None

    return domain.comm.bcast(out, root=0)
