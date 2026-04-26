import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _make_mesh(comm, n):
    return mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)


def _boundary_facets(domain):
    fdim = domain.topology.dim - 1
    return mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))


def _source_expr(x):
    return ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1]) + 0.3 * ufl.sin(12.0 * ufl.pi * x[0]) * ufl.sin(10.0 * ufl.pi * x[1])


def _tau_supg(h, beta_norm, eps):
    # Robust algebraic SUPG parameter for stationary convection-diffusion
    # tau ~ 1 / sqrt((2|beta|/h)^2 + (C eps / h^2)^2)
    return 1.0 / ufl.sqrt((2.0 * beta_norm / h) ** 2 + (9.0 * 4.0 * eps / (h * h)) ** 2)


def _solve_single(comm, n, degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    domain = _make_mesh(comm, n)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps_val = 0.01
    beta_vec = np.array([12.0, 6.0], dtype=np.float64)
    beta = fem.Constant(domain, ScalarType(beta_vec))
    f_expr = _source_expr(x)

    h = ufl.CellDiameter(domain)
    beta_norm = np.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau = _tau_supg(h, beta_norm, eps_val)

    Lu = -eps_val * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    Lv = ufl.dot(beta, ufl.grad(v))

    a = (
        eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * Lv * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * Lv * ufl.dx

    facets = _boundary_facets(domain)
    fdim = domain.topology.dim - 1
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=20000)
    if ksp_type.lower() == "gmres":
        solver.setGMRESRestart(200)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solve failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1)
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

    iterations = int(solver.getIterationNumber())

    # Residual-based verification: ||Lu-f||_L2 approximated weakly in DG0 projection-free style
    V0 = fem.functionspace(domain, ("DG", 0))
    w0 = ufl.TestFunction(V0)
    cell_vol = fem.form(ufl.inner(ufl.Constant(domain, ScalarType(1.0)), w0) * ufl.dx)

    strong_res = -eps_val * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - f_expr
    res_sq = fem.assemble_scalar(fem.form(strong_res * strong_res * ufl.dx))
    res_sq = comm.allreduce(res_sq, op=MPI.SUM)
    residual_l2 = float(np.sqrt(abs(res_sq)))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": iterations,
        "residual_l2": residual_l2,
    }
    return domain, V, uh, info


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    owners = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            owners.append(i)

    if len(points_on_proc) > 0:
        evals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        vals[np.array(owners, dtype=np.int32)] = np.real(evals)

    # Parallel-safe gather by replacing NaNs where available on owning proc
    comm = domain.comm
    gathered = comm.allgather(vals)
    out = np.full_like(vals, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        out[mask] = arr[mask]

    # Boundary points should belong to some process; still guard against tiny geometry misses
    if np.isnan(out).any():
        out[np.isnan(out)] = 0.0

    return out.reshape(ny, nx)


def _coarsen_compare(comm, fine_grid, coarse_grid):
    diff = fine_grid - coarse_grid
    loc = np.mean(diff ** 2)
    glob = comm.allreduce(loc, op=MPI.SUM) / comm.size
    return float(np.sqrt(glob))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]

    # Accuracy/time adaptive strategy:
    # start with stabilized P1 on reasonably fine mesh; if cheap enough, refine once.
    candidates = [(72, 1), (96, 1), (120, 1)]
    time_budget = 42.846
    chosen = None
    chosen_grid = None
    coarse_grid = None

    for idx, (n, degree) in enumerate(candidates):
        domain, V, uh, info = _solve_single(comm, n, degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9)
        fine_grid = _sample_on_grid(domain, uh, grid)

        elapsed = time.perf_counter() - t0
        if idx == 0:
            chosen = (domain, V, uh, info)
            chosen_grid = fine_grid
            coarse_grid = fine_grid
            if elapsed > 0.65 * time_budget:
                break
            continue

        # mesh convergence style verification on evaluator grid
        err_grid = _coarsen_compare(comm, fine_grid, coarse_grid)
        info["grid_change_l2"] = err_grid

        chosen = (domain, V, uh, info)
        chosen_grid = fine_grid
        coarse_grid = fine_grid

        # If already sufficiently resolved or nearing budget, stop
        if err_grid < 8e-3 or elapsed > 0.82 * time_budget:
            break

    domain, V, uh, info = chosen

    result = {
        "u": chosen_grid,
        "solver_info": {
            "mesh_resolution": int(info["mesh_resolution"]),
            "element_degree": int(info["element_degree"]),
            "ksp_type": str(info["ksp_type"]),
            "pc_type": str(info["pc_type"]),
            "rtol": float(info["rtol"]),
            "iterations": int(info["iterations"]),
            "residual_l2": float(info.get("residual_l2", np.nan)),
            "grid_change_l2": float(info.get("grid_change_l2", np.nan)),
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }
    return result
