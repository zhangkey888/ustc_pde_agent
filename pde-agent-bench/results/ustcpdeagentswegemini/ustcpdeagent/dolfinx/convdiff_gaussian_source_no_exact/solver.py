import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _source_expr(x):
    return np.exp(-250.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))


def _build_problem(n, degree=1, epsilon=0.02, beta=(8.0, 3.0), supg=True):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_c = fem.Constant(domain, np.array(beta, dtype=np.float64))
    f = fem.Function(V)
    f.interpolate(_source_expr)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx

    if supg:
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.inner(beta_c, beta_c) + 1.0e-16)
        tau = h / (2.0 * beta_norm)
        r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.inner(beta_c, ufl.grad(u))
        r_L = f
        a += tau * ufl.inner(beta_c, ufl.grad(v)) * r_u * ufl.dx
        L += tau * ufl.inner(beta_c, ufl.grad(v)) * r_L * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    return domain, V, a, L, bc, f


def _sample_on_grid(u_fun, bbox, nx, ny):
    domain = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx.append(i)

    if len(points_on_proc) > 0:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32)).reshape(-1)
        local_vals[np.array(idx, dtype=np.int32)] = np.real(vals)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape((ny, nx))
    return None


def _compute_metrics(domain, uh, f_fun, epsilon=0.02, beta=(8.0, 3.0)):
    V = uh.function_space
    W = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(W)
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_c = fem.Constant(domain, np.array(beta, dtype=np.float64))
    h = ufl.CellDiameter(domain)

    strong_res = -eps_c * ufl.div(ufl.grad(uh)) + ufl.inner(beta_c, ufl.grad(uh)) - f_fun
    cell_res = fem.assemble_scalar(fem.form(ufl.inner(strong_res, strong_res) * h * h * ufl.dx))
    l2u = fem.assemble_scalar(fem.form(uh * uh * ufl.dx))
    seminorm = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))

    comm = domain.comm
    cell_res = comm.allreduce(cell_res, op=MPI.SUM)
    l2u = comm.allreduce(l2u, op=MPI.SUM)
    seminorm = comm.allreduce(seminorm, op=MPI.SUM)
    return {
        "residual_indicator": float(np.sqrt(max(cell_res, 0.0))),
        "l2_norm": float(np.sqrt(max(l2u, 0.0))),
        "h1_seminorm": float(np.sqrt(max(seminorm, 0.0))),
    }


def _solve_single(n, degree, epsilon, beta, ksp_type, pc_type, rtol, supg=True):
    domain, V, a, L, bc, f_fun = _build_problem(n=n, degree=degree, epsilon=epsilon, beta=beta, supg=supg)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    uh = fem.Function(V)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(A)
        solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    its = int(solver.getIterationNumber())
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": its,
        "solve_time": elapsed,
    }
    metrics = _compute_metrics(domain, uh, f_fun, epsilon=epsilon, beta=beta)
    return domain, uh, info, metrics


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    output = case_spec["output"]["grid"]
    nx = int(output["nx"])
    ny = int(output["ny"])
    bbox = output["bbox"]

    epsilon = float(pde.get("epsilon", 0.02))
    beta = tuple(float(v) for v in pde.get("beta", [8.0, 3.0]))
    time_budget = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 30.0)))

    degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9

    candidates = [72, 96, 128, 160]
    if degree >= 2:
        candidates = [64, 96, 128, 160]

    start = time.perf_counter()
    best = None
    history = []

    for n in candidates:
        domain, uh, info, metrics = _solve_single(
            n=n, degree=degree, epsilon=epsilon, beta=beta,
            ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, supg=True
        )
        total_elapsed = time.perf_counter() - start
        score = metrics["residual_indicator"]
        history.append((n, info, metrics, total_elapsed))
        best = (domain, uh, info, metrics)
        if total_elapsed > 0.55 * time_budget and n >= candidates[1]:
            break

    domain, uh, info, metrics = best

    if time.perf_counter() - start < 0.35 * time_budget:
        extra_n = min(max(info["mesh_resolution"] + 32, info["mesh_resolution"]), 224)
        domain2, uh2, info2, metrics2 = _solve_single(
            n=extra_n, degree=degree, epsilon=epsilon, beta=beta,
            ksp_type=ksp_type, pc_type=pc_type, rtol=rtol, supg=True
        )
        if metrics2["residual_indicator"] <= metrics["residual_indicator"] or info2["solve_time"] < 0.5 * time_budget:
            domain, uh, info, metrics = domain2, uh2, info2, metrics2

    u_grid = _sample_on_grid(uh, bbox, nx, ny)
    if domain.comm.rank != 0:
        return {"u": None, "solver_info": {}}

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
        "verification": {
            "type": "mesh_residual_indicator",
            "residual_indicator": metrics["residual_indicator"],
            "l2_norm": metrics["l2_norm"],
            "h1_seminorm": metrics["h1_seminorm"],
            "history": [
                {
                    "mesh_resolution": int(nh),
                    "residual_indicator": float(mh["residual_indicator"]),
                    "solve_time": float(ih["solve_time"]),
                }
                for nh, ih, mh, _ in history
            ],
        },
    }
    return {"u": u_grid, "solver_info": solver_info}
