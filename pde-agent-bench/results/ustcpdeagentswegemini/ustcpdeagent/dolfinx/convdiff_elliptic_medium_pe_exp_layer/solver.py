import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function_on_grid(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    if domain.comm.size > 1:
        recv = np.empty_like(values)
        domain.comm.Allreduce(values, recv, op=MPI.MAX)
        values = recv

    return values.reshape((ny, nx))


def _build_and_solve(n=80, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(0.05)
    beta_vec = np.array([4.0, 0.0], dtype=np.float64)
    beta = fem.Constant(domain, ScalarType(beta_vec))
    h = ufl.CellDiameter(domain)

    u_exact_ufl = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    lap_u_exact = ufl.div(grad_u_exact)
    f_ufl = -eps * lap_u_exact + ufl.dot(beta, grad_u_exact)

    uD = fem.Function(V)
    uD.interpolate(lambda X: np.exp(2.0 * X[0]) * np.sin(np.pi * X[1]))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    beta_norm = float(np.linalg.norm(beta_vec))
    tau = (h / (2.0 * beta_norm)) * (ufl.cosh(beta_norm * h / (2.0 * eps)) / ufl.sinh(beta_norm * h / (2.0 * eps)) - 2.0 * eps / (beta_norm * h))
    tau = ufl.max_value(tau, 0.0)

    Lu = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    Lv = ufl.dot(beta, ufl.grad(v))

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * Lv * ufl.dx
    )
    L = f_ufl * v * ufl.dx + tau * f_ufl * Lv * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)
    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP did not converge, reason={reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(lambda X: np.exp(2.0 * X[0]) * np.sin(np.pi * X[1]))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact_fun.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    linf_local = np.max(np.abs(e.x.array)) if e.x.array.size > 0 else 0.0
    linf_error = comm.allreduce(linf_local, op=MPI.MAX)

    its = solver.getIterationNumber()
    return {
        "domain": domain,
        "uh": uh,
        "l2_error": float(l2_error),
        "linf_error": float(linf_error),
        "iterations": int(its),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": sampled solution array of shape (ny, nx)
    - "solver_info": metadata including accuracy verification fields
    """
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    candidates = [
        (72, 1),
        (88, 1),
        (104, 1),
        (72, 2),
    ]

    best = None
    for n, degree in candidates:
        if time.perf_counter() - t0 > 5.4:
            break
        result = _build_and_solve(n=n, degree=degree, ksp_type="gmres", pc_type="ilu", rtol=1e-10)
        best = result
        if result["l2_error"] <= 2.84e-4 and (time.perf_counter() - t0) < 5.0:
            continue

    if best is None:
        best = _build_and_solve(n=80, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-10)

    u_grid = _probe_function_on_grid(best["domain"], best["uh"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "verification_l2_error": best["l2_error"],
        "verification_linf_error": best["linf_error"],
        "wall_time_sec": time.perf_counter() - t0,
        "stabilization": "SUPG",
        "case_id": "convdiff_elliptic_medium_pe_exp_layer",
    }

    return {"u": u_grid, "solver_info": solver_info}
