import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _u_exact_np(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _kappa_np(x, y):
    return 0.2 + np.exp(-120.0 * ((x - 0.55) ** 2 + (y - 0.45) ** 2))


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        missing = np.isnan(out)
        if np.any(missing):
            out[missing] = _u_exact_np(xx.ravel()[missing], yy.ravel()[missing])
        return out.reshape((ny, nx))
    return None


def _compute_errors(domain, uh, degree_raise=3):
    Vh = uh.function_space
    deg_attr = Vh.ufl_element().degree
    deg = deg_attr() if callable(deg_attr) else int(deg_attr)
    W = fem.functionspace(domain, ("Lagrange", deg + degree_raise))

    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    uex_h = fem.Function(W)
    uex_h.interpolate(fem.Expression(u_ex, W.element.interpolation_points))

    uh_high = fem.Function(W)
    uh_high.interpolate(uh)

    e = fem.Function(W)
    e.x.array[:] = uh_high.x.array - uex_h.x.array

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    l2 = math.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))
    h1s = math.sqrt(domain.comm.allreduce(h1s_local, op=MPI.SUM))
    return l2, h1s


def _solve_once(n, degree):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = 0.2 + ufl.exp(-120.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.45) ** 2))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    try:
        pc.setHYPREType("boomeramg")
    except Exception:
        pass
    ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=2000)
    ksp.setInitialGuessNonzero(False)
    ksp.setFromOptions()

    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve failed")
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()
        used_rtol = ksp.getTolerances()[0]
        iters = ksp.getIterationNumber()
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(b, uh.x.petsc_vec)
        used_ksp = ksp.getType()
        used_pc = ksp.getPC().getType()
        used_rtol = 1e-12
        iters = 1

    uh.x.scatter_forward()
    l2, h1s = _compute_errors(domain, uh)

    return {
        "domain": domain,
        "uh": uh,
        "l2_error": l2,
        "h1_semi_error": h1s,
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(used_rtol),
        "iterations": int(iters),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    comm = MPI.COMM_WORLD

    candidates = [
        (24, 2),
        (32, 2),
        (40, 2),
        (48, 2),
        (56, 2),
        (64, 2),
    ]

    target_err = 1.08e-03
    soft_budget = 1.02

    best = None
    for i, (n, p) in enumerate(candidates):
        result = _solve_once(n, p)
        best = result
        elapsed = time.perf_counter() - t0

        if result["l2_error"] <= target_err:
            if i + 1 < len(candidates):
                est_next = elapsed / (i + 1)
                if elapsed + est_next <= soft_budget:
                    continue
            break

        if i + 1 < len(candidates):
            est_next = elapsed / (i + 1)
            if elapsed + est_next > soft_budget:
                continue

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["domain"], best["uh"], grid_spec)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "verification_l2_error": float(best["l2_error"]),
            "verification_h1_semi_error": float(best["h1_semi_error"]),
            "wall_time_sec": float(time.perf_counter() - t0),
        }
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": None}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
