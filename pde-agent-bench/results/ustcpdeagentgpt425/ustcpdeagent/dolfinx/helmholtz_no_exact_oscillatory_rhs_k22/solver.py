import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_denominator(k: float) -> float:
    return (10.0 * np.pi) ** 2 + (8.0 * np.pi) ** 2 - k ** 2


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.array(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        vals[np.isnan(vals)] = 0.0
        grid = vals.reshape(ny, nx)
    else:
        grid = None

    grid = domain.comm.bcast(grid, root=0)
    return np.asarray(grid, dtype=np.float64)


def _solve_once(n, degree, k, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(8.0 * ufl.pi * x[1])
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    used_ksp = "gmres"
    used_pc = "ilu"

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"helm_{n}_{degree}_",
            petsc_options={
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": rtol,
                "ksp_atol": 1e-12,
                "ksp_max_it": 5000,
            },
        )
        uh = problem.solve()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        if int(ksp.getConvergedReason()) < 0:
            raise RuntimeError("iterative solve diverged")
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"helm_lu_{n}_{degree}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        used_ksp = "preonly"
        used_pc = "lu"

    uh.x.scatter_forward()

    denom = _exact_denominator(k)
    u_exact = f_expr / denom
    err_form = fem.form((uh - u_exact) * (uh - u_exact) * ufl.dx)
    l2_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

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

    r = b.copy()
    A.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    rel_res = r.norm() / max(b.norm(), 1e-30)

    return domain, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error_exact": float(l2_error),
        "relative_residual": float(rel_res),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    k = float(pde.get("k", case_spec.get("wavenumber", 22.0)))
    rtol = 1e-9

    candidates = [(64, 2), (96, 2), (128, 2)]
    best_u = None
    best_info = None
    prev_grid = None

    for i, (n, degree) in enumerate(candidates):
        domain, uh, info = _solve_once(n, degree, k, rtol)
        u_grid = _sample_on_grid(domain, uh, grid_spec)

        if prev_grid is not None:
            denom = max(np.linalg.norm(u_grid), 1e-30)
            info["grid_relative_change_vs_prev"] = float(np.linalg.norm(u_grid - prev_grid) / denom)
        else:
            info["grid_relative_change_vs_prev"] = None

        best_u = u_grid
        best_info = info
        prev_grid = u_grid

        elapsed = time.perf_counter() - t0
        if i < len(candidates) - 1:
            if elapsed > 20.0:
                break
            if (
                info["l2_error_exact"] < 2e-2
                and info["relative_residual"] < 1e-9
                and info["grid_relative_change_vs_prev"] is not None
                and info["grid_relative_change_vs_prev"] < 5e-3
            ):
                break

    solver_info = {
        "mesh_resolution": int(best_info["mesh_resolution"]),
        "element_degree": int(best_info["element_degree"]),
        "ksp_type": str(best_info["ksp_type"]),
        "pc_type": str(best_info["pc_type"]),
        "rtol": float(best_info["rtol"]),
        "iterations": int(best_info["iterations"]),
    }

    return {"u": np.asarray(best_u, dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 22.0},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
