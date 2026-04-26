import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _build_and_solve(n, degree=1, k=15.0, comm=MPI.COMM_WORLD,
                     ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-12,
        "ksp_max_it": 5000,
    }
    if ksp_type == "gmres":
        opts["ksp_gmres_restart"] = 200
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    uh = fem.Function(V)
    problem = petsc.LinearProblem(
        a, L, u=uh, bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=f"helmholtz_{n}_{degree}_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    reason = int(ksp.getConvergedReason())

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    r = b.copy()
    A.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    res_norm = r.norm()
    b_norm = b.norm()
    rel_res = res_norm / b_norm if b_norm > 0 else res_norm

    return {
        "domain": domain,
        "u": uh,
        "iterations": its,
        "reason": reason,
        "rel_residual": float(rel_res),
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
    }


def _sample_function_on_grid(domain, ufun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals_local = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        values = ufun.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals_local[np.array(ids, dtype=np.int32)] = np.asarray(values).reshape(-1).real

    gathered = domain.comm.allgather(vals_local)
    vals = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(vals) & ~np.isnan(arr)
        vals[mask] = arr[mask]

    vals = np.nan_to_num(vals, nan=0.0)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    pde = case_spec.get("pde", {})
    grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    k = float(pde.get("wavenumber", 15.0))

    degree = 2
    candidates = [(72, "gmres", "ilu"), (96, "gmres", "ilu"), (120, "preonly", "lu")]
    chosen = None
    for n, ksp_type, pc_type in candidates:
        try:
            chosen = _build_and_solve(n=n, degree=degree, k=k, ksp_type=ksp_type, pc_type=pc_type, rtol=1e-9)
            if time.perf_counter() - t0 > 28.0:
                break
        except Exception:
            continue

    if chosen is None:
        chosen = _build_and_solve(n=64, degree=1, k=k, ksp_type="preonly", pc_type="lu", rtol=1e-10)

    verification = {}
    if time.perf_counter() - t0 < 20.0:
        try:
            n_ref = min(int(chosen["mesh_resolution"] * 4 / 3), 128)
            ref = _build_and_solve(n=n_ref, degree=chosen["element_degree"], k=k,
                                   ksp_type="gmres", pc_type="ilu", rtol=1e-10)
            u_grid0 = _sample_function_on_grid(chosen["domain"], chosen["u"], grid)
            u_grid1 = _sample_function_on_grid(ref["domain"], ref["u"], grid)
            diff = u_grid1 - u_grid0
            verification["grid_refinement_l2"] = float(np.sqrt(np.mean(diff ** 2)))
            verification["grid_refinement_linf"] = float(np.max(np.abs(diff)))
            verification["ref_mesh_resolution"] = int(ref["mesh_resolution"])
            if time.perf_counter() - t0 < 38.0 and verification["grid_refinement_l2"] > 1e-3:
                chosen = ref
        except Exception:
            pass

    u_grid = _sample_function_on_grid(chosen["domain"], chosen["u"], grid)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(chosen["element_degree"]),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
    }
    solver_info.update(verification)
    solver_info["relative_residual_verification"] = float(chosen["rel_residual"])
    solver_info["linear_converged_reason"] = int(chosen["reason"])

    return {"u": u_grid, "solver_info": solver_info}
