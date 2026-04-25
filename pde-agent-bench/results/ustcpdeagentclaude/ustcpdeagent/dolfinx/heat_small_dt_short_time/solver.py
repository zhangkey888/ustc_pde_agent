import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    t0 = 0.0
    t_end = 0.06
    dt_val = 0.003
    kappa_val = 1.0

    # Try to read from case_spec
    try:
        time_spec = case_spec.get("pde", {}).get("time", {})
        t0 = float(time_spec.get("t0", t0))
        t_end = float(time_spec.get("t_end", t_end))
        dt_val = float(time_spec.get("dt", dt_val))
    except Exception:
        pass

    # Spatial discretization
    N = 128
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))

    # Manufactured solution: u = exp(-t)*sin(4*pi*x)*sin(4*pi*y)
    # du/dt = -exp(-t)*sin*sin
    # -lap(u) = (32*pi^2)*exp(-t)*sin*sin
    # f = du/dt - kappa*lap(u) = (-1 + kappa*32*pi^2)*exp(-t)*sin*sin
    def u_exact_expr(t_c):
        return ufl.exp(-t_c) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])

    def f_expr(t_c):
        return (-1.0 + kappa_val * 32.0 * ufl.pi * ufl.pi) * ufl.exp(-t_c) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(u_exact_expr(fem.Constant(domain, PETSc.ScalarType(t0))),
                                  V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    # Store initial for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Boundary condition (time-dependent from exact solution)
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_expr(t_const), V.element.interpolation_points)
    u_bc.interpolate(bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form: backward Euler
    # (u - u_n)/dt - kappa*lap(u) = f(t^{n+1})
    # => (u, v) + dt*kappa*(grad(u), grad(v)) = (u_n, v) + dt*(f, v)
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = u_tr * v * ufl.dx + dt_const * kappa * ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx + dt_const * f_expr(t_const) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    rtol = 1e-10
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)

    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur = t0 + (step + 1) * dt_val
        t_const.value = t_cur
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

    # Verify against exact
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_exact_expr(fem.Constant(domain, PETSc.ScalarType(t_end))),
                                          V.element.interpolation_points))
    err_local = fem.assemble_scalar(fem.form((u_sol - u_ex_func) ** 2 * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    print(f"L2 error at t={t_end}: {err_L2:.6e}, steps={n_steps}, iters={total_iters}")

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

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

    u_grid_flat = np.zeros(nx * ny)
    u_init_flat = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_flat[eval_map] = vals_init.flatten()

    u_grid = u_grid_flat.reshape(ny, nx)
    u_initial_grid = u_init_flat.reshape(ny, nx)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.06, "dt": 0.003}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
