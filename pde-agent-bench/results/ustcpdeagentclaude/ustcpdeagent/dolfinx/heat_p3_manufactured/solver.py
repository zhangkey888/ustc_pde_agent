import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Time params
    pde = case_spec.get("pde", {})
    time_cfg = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(time_cfg.get("t0", 0.0))
    t_end = float(time_cfg.get("t_end", 0.08))
    dt_suggested = float(time_cfg.get("dt", 0.008))

    # Grid config
    out = case_spec.get("output", {})
    grid = out.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    # Discretization
    degree = 3
    N = 32  # mesh resolution; with P3 should give high accuracy

    # Time stepping: use small dt for accuracy with backward Euler
    # Use dt = 0.001 => 80 steps
    n_steps = 80
    dt_val = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Exact solution and source
    # u = exp(-t) sin(pi x) sin(2 pi y)
    # u_t = -exp(-t) sin(pi x) sin(2 pi y)
    # Delta u = -(pi^2 + 4 pi^2) exp(-t) sin(pi x) sin(2 pi y) = -5 pi^2 * u
    # f = u_t - kappa * Delta u = (-1 + 5 pi^2) * u
    kappa = 1.0
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f_ufl = (-1.0 + 5.0 * ufl.pi**2 * kappa) * u_exact_ufl

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(
        ufl.exp(-fem.Constant(domain, PETSc.ScalarType(t0))) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        V.element.interpolation_points
    )
    u_n.interpolate(u_init_expr)

    # Boundary condition (time-dependent)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form: Backward Euler
    # (u - u_n)/dt - kappa * div(grad u) = f
    # u * v + dt * kappa * grad u . grad v = (u_n + dt*f) * v
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = u_tr * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = (u_n + dt_const * f_ufl) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)
    total_iters = 0

    t = t0
    for step in range(n_steps):
        t_new = t + dt_val
        t_const.value = t_new  # for RHS f and BC at new time
        # update BC
        u_bc.interpolate(bc_expr)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_values[idx] = vals[k, 0] if vals.ndim > 1 else vals[k]

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial field
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(
        ufl.exp(-fem.Constant(domain, PETSc.ScalarType(0.0))) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        V.element.interpolation_points
    ))
    u_init_vals = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals0 = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_init_vals[idx] = vals0[k, 0] if vals0.ndim > 1 else vals0[k]
    u_init_grid = u_init_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": int(total_iters),
            "dt": float(dt_val),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.08, "dt": 0.008}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    res = solve(case)
    elapsed = time.time() - t0
    u = res["u"]
    # Compute error
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.08) * np.sin(np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((u - u_exact) ** 2))
    max_err = np.max(np.abs(u - u_exact))
    print(f"Time: {elapsed:.2f}s, L2 err: {err:.3e}, max err: {max_err:.3e}")
    print("solver_info:", res["solver_info"])
