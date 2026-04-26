import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _build_problem(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    kappa_ufl = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5) ** 2)
    f_ufl = 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    if ksp_type == "cg":
        try:
            solver.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
        except Exception:
            pass

    solver.setFromOptions()

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    reason = solver.getConvergedReason()
    if reason <= 0:
        solver.destroy()
        A.destroy()
        b.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        used_ksp = "preonly"
        used_pc = "lu"
    else:
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()

    its = solver.getIterationNumber()
    return domain, V, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": int(its),
    }


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
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells, dtype=np.int32))
        values[np.asarray(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr)
        merged[mask] = arr[mask]

    if np.any(~np.isfinite(merged)):
        merged[~np.isfinite(merged)] = 0.0

    return merged.reshape(ny, nx)


def _evaluate_at_points(domain, uh, pts):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells, dtype=np.int32))
        values[np.asarray(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isfinite(arr)
        merged[mask] = arr[mask]
    return merged


def _estimate_discretization_error(domain_c, uh_c, domain_f, uh_f):
    # Probe coarse solution against fine solution on an interior tensor grid.
    m = 41
    xs = np.linspace(0.0, 1.0, m)
    ys = np.linspace(0.0, 1.0, m)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(m * m, dtype=np.float64)])

    uc = _evaluate_at_points(domain_c, uh_c, pts)
    uf = _evaluate_at_points(domain_f, uh_f, pts)

    mask = np.isfinite(uc) & np.isfinite(uf)
    if not np.any(mask):
        return float("nan")
    diff = uc[mask] - uf[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]

    # Adaptive accuracy/time trade-off for <= ~5.17 s budget:
    # start with accurate quadratic FEM; refine if runtime permits.
    degree = 2
    candidates = [56, 72, 88]
    chosen = None
    chosen_meta = None
    chosen_domain = None
    chosen_u = None
    verification = {}

    for i, n in enumerate(candidates):
        domain, V, uh, meta = _build_problem(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        elapsed = time.perf_counter() - t0

        # perform one-step verification if possible
        if i == 0:
            # Compare with one refined solve if enough budget remains
            if elapsed < 2.2:
                nf = min(2 * n, 112)
                domain_f, V_f, uh_f, meta_f = _build_problem(n=nf, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
                err_est = _estimate_discretization_error(domain, uh, domain_f, uh_f)
                verification = {
                    "verification_type": "mesh_refinement_probe_l2",
                    "reference_mesh_resolution": int(nf),
                    "estimated_error": float(err_est),
                }
                chosen = nf
                chosen_meta = meta_f
                chosen_domain = domain_f
                chosen_u = uh_f
                elapsed = time.perf_counter() - t0
                if elapsed > 4.2:
                    break
                continue
            else:
                verification = {
                    "verification_type": "mesh_refinement_probe_l2",
                    "reference_mesh_resolution": None,
                    "estimated_error": None,
                }

        chosen = n
        chosen_meta = meta
        chosen_domain = domain
        chosen_u = uh

        if elapsed > 4.2:
            break

    if chosen_u is None:
        chosen_domain, _, chosen_u, chosen_meta = _build_problem(n=64, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10)

    u_grid = _sample_on_grid(chosen_domain, chosen_u, grid_spec)

    solver_info = dict(chosen_meta)
    solver_info.update(verification)

    return {"u": u_grid, "solver_info": solver_info}
