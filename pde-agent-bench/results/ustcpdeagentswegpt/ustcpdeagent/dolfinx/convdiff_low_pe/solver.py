import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids_on_proc = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        values[np.array(ids_on_proc, dtype=np.int32)] = np.real(vals)

    gathered = domain.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    if np.isnan(merged).any():
        raise RuntimeError("Failed to sample FEM solution on requested output grid")

    return merged.reshape(ny, nx)


def _build_and_solve(n, degree, rtol=1e-10, use_supg=True):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    eps = 0.2
    beta_np = np.array([1.0, 0.5], dtype=np.float64)
    beta = fem.Constant(domain, np.array(beta_np, dtype=ScalarType))

    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = -eps * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f_ufl * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        tau = h / (2.0 * beta_norm + 4.0 * eps / h)
        r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        a += tau * r_trial * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * f_ufl * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

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
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=2000)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    try:
        ksp.solve(b, uh.x.petsc_vec)
    except Exception:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: np.sin(pi * X[0]) * np.sin(pi * X[1]))
    err_l2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx)), op=MPI.SUM))

    return {
        "domain": domain,
        "uh": uh,
        "err_l2": float(err_l2),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "elapsed": float(elapsed),
    }


def solve(case_spec: dict) -> dict:
    wall_start = time.perf_counter()
    budget = 0.966

    candidates = [(20, 1), (28, 1), (36, 1), (24, 2), (32, 2)]
    best = None

    for n, degree in candidates:
        if time.perf_counter() - wall_start > 0.8 * budget:
            break
        try:
            current = _build_and_solve(n=n, degree=degree, rtol=1e-10, use_supg=True)
        except Exception:
            continue
        best = current
        if current["err_l2"] <= 1.63e-3 and (time.perf_counter() - wall_start) > 0.65 * budget:
            break

    if best is None:
        best = _build_and_solve(n=24, degree=1, rtol=1e-9, use_supg=True)

    if best["err_l2"] > 1.63e-3 and time.perf_counter() - wall_start < 0.6 * budget:
        for n, degree in [(40, 2), (48, 2)]:
            try:
                cand = _build_and_solve(n=n, degree=degree, rtol=1e-11, use_supg=True)
            except Exception:
                continue
            best = cand
            if best["err_l2"] <= 1.63e-3:
                break

    u_grid = _sample_function_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "verification_L2_error": best["err_l2"],
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
