import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func, domain, grid_spec):
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

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.real(np.asarray(vals).reshape(-1))

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]
    global_vals[np.isnan(global_vals)] = 0.0
    return global_vals.reshape((ny, nx))


def _rhs_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.cos(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])


def _solve_level(n, k, degree=1, rtol=1e-9, ksp_type="gmres", pc_type="ilu"):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = _rhs_expr(domain)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k * k) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

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
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=5000)

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("GMRES/ILU did not converge")
        actual_ksp = ksp_type
        actual_pc = pc_type
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        actual_ksp = "preonly"
        actual_pc = "lu"

    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())
    if actual_ksp == "preonly":
        iterations = 1

    return {
        "domain": domain,
        "u": uh,
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "rtol": float(rtol),
        "iterations": iterations,
    }


def _grid_l2_diff(sol_a, sol_b, nx=81, ny=81):
    grid = {"nx": nx, "ny": ny, "bbox": [0.0, 1.0, 0.0, 1.0]}
    ua = _sample_function_on_grid(sol_a["u"], sol_a["domain"], grid)
    ub = _sample_function_on_grid(sol_b["u"], sol_b["domain"], grid)
    return float(np.sqrt(np.mean((ub - ua) ** 2)))


def solve(case_spec: dict) -> dict:
    t0 = time.time()
    pde = case_spec.get("pde", {})
    k = float(pde.get("k", case_spec.get("k", 8.0)))
    grid_spec = case_spec["output"]["grid"]

    degree = 1
    rtol = 1e-9
    levels = [64, 96, 128]
    results = []

    prev = None
    for n in levels:
        cur = _solve_level(n=n, k=k, degree=degree, rtol=rtol, ksp_type="gmres", pc_type="ilu")
        results.append(cur)
        if prev is not None:
            err_est = _grid_l2_diff(prev, cur, nx=min(81, int(grid_spec["nx"])), ny=min(81, int(grid_spec["ny"])))
            cur["error_estimate_vs_prev"] = err_est
            elapsed = time.time() - t0
            if err_est < 8e-3 and elapsed > 2.0:
                break
            if elapsed > 35.0:
                break
        prev = cur

    best = results[-1]
    u_grid = _sample_function_on_grid(best["u"], best["domain"], grid_spec)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": int(sum(r["iterations"] for r in results)),
        "accuracy_check": {
            "type": "mesh_convergence",
            "levels_tested": [r["mesh_resolution"] for r in results],
        },
    }
    if "error_estimate_vs_prev" in best:
        solver_info["accuracy_check"]["estimated_L2_grid_diff"] = best["error_estimate_vs_prev"]

    return {"u": u_grid, "solver_info": solver_info}
