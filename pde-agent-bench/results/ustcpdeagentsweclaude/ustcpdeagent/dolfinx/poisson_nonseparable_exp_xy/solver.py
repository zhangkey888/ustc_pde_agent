import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

COMM = MPI.COMM_WORLD


def _u_exact_numpy(x):
    return np.exp(x[0] * x[1]) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _manufactured_rhs_ufl(x):
    uex = ufl.exp(x[0] * x[1]) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    return -ufl.div(ufl.grad(uex))


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    local_pts = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_pts.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_pts:
        arr = u_func.eval(np.array(local_pts, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals[np.array(local_ids, dtype=np.int32)] = np.asarray(arr).reshape(-1)

    gathered = COMM.allgather(vals)
    out = np.full(points.shape[0], np.nan, dtype=np.float64)
    for g in gathered:
        mask = ~np.isnan(g)
        out[mask] = g[mask]
    return out


def _compute_errors(u_h):
    msh = u_h.function_space.mesh
    x = ufl.SpatialCoordinate(msh)
    uex = ufl.exp(x[0] * x[1]) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    err_l2_form = fem.form((u_h - uex) ** 2 * ufl.dx)
    err_h1_form = fem.form(ufl.inner(ufl.grad(u_h - uex), ufl.grad(u_h - uex)) * ufl.dx)
    l2 = np.sqrt(COMM.allreduce(fem.assemble_scalar(err_l2_form), op=MPI.SUM))
    h1 = np.sqrt(COMM.allreduce(fem.assemble_scalar(err_h1_form), op=MPI.SUM))
    return float(l2), float(h1)


def _solve_once(n, p, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    msh = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", p))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = _manufactured_rhs_ufl(x) * v * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(COMM)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    if ksp_type == "cg":
        try:
            solver.getPC().setHYPREType("boomeramg")
        except Exception:
            pass

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve failed")
        ksp_used = solver.getType()
        pc_used = solver.getPC().getType()
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(COMM)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_used = "preonly"
        pc_used = "lu"

    iterations = int(solver.getIterationNumber())
    l2_error, h1_error = _compute_errors(uh)

    return {
        "u_h": uh,
        "mesh_resolution": int(n),
        "element_degree": int(p),
        "ksp_type": str(ksp_used),
        "pc_type": str(pc_used),
        "rtol": float(rtol),
        "iterations": iterations,
        "L2_error": l2_error,
        "H1_semi_error": h1_error,
    }


def _sample_to_grid(u_h, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    values = _probe_function(u_h, points)
    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    candidates = [(24, 2), (32, 2), (40, 2), (48, 2), (56, 2), (64, 2)]
    budget = 1.786
    safety = 0.20
    best = None

    for n, p in candidates:
        t_start = time.perf_counter()
        current = _solve_once(n, p, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        elapsed = time.perf_counter() - t_start
        best = current
        if (time.perf_counter() - t0) + max(elapsed, 0.10) > budget - safety:
            break

    u_grid = _sample_to_grid(best["u_h"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "L2_error": best["L2_error"],
        "H1_semi_error": best["H1_semi_error"],
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
