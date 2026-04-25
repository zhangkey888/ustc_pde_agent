import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def _probe_function(u_func, points):
    """
    Evaluate scalar fem.Function at points of shape (N, 3).
    Returns array of shape (N,), with NaN for points not found on this rank.
    """
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            idx_local.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64),
                          np.array(cells_local, dtype=np.int32))
        values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals_local = _probe_function(u_func, pts)
    comm = u_func.function_space.mesh.comm
    vals_all = comm.allreduce(np.nan_to_num(vals_local, nan=0.0), op=MPI.SUM)

    return vals_all.reshape((ny, nx))


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = np.pi

    u_exact_ufl = ufl.sin(2.0 * pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    kappa_ufl = 1.0 + 0.3 * ufl.sin(8.0 * pi * x[0]) * ufl.sin(8.0 * pi * x[1])
    f_ufl = -ufl.div(kappa_ufl * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

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
        uh.x.scatter_forward()
    except Exception:
        solver.destroy()
        A.destroy()
        b.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    ksp_type_actual = solver.getType()
    pc_type_actual = solver.getPC().getType()
    iterations = int(solver.getIterationNumber())

    ue = fem.Function(V)
    ue.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    err_local = fem.assemble_scalar(fem.form((uh - ue) ** 2 * ufl.dx))
    norm_local = fem.assemble_scalar(fem.form(ue ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
    rel_l2 = err_l2 / norm_l2 if norm_l2 > 0 else err_l2

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type_actual),
        "pc_type": str(pc_type_actual),
        "rtol": float(rtol),
        "iterations": iterations,
        "l2_error": float(err_l2),
        "relative_l2_error": float(rel_l2),
        "solve_time": float(solve_time),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    """
    Solve variable-coefficient Poisson problem on the unit square and sample
    the solution onto the requested uniform grid.
    """
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    # Conservative defaults, then adapt upward if ample time budget remains.
    candidates = [
        (48, 1),
        (64, 1),
        (48, 2),
        (64, 2),
        (80, 2),
    ]

    best_u = None
    best_info = None

    budget = 3.438
    safety = 0.55

    for idx, (n, degree) in enumerate(candidates):
        try:
            uh, info = _solve_once(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        except Exception:
            uh, info = _solve_once(n=n, degree=degree, ksp_type="preonly", pc_type="lu", rtol=1e-10)

        elapsed = time.perf_counter() - t_start
        best_u, best_info = uh, info

        # Stop if we're already accurate enough and moving higher likely risks budget.
        if info["l2_error"] <= 3.55e-03:
            if idx + 1 < len(candidates):
                next_est = max(info["solve_time"] * 2.0, 0.15)
                if elapsed + next_est > safety * budget:
                    break
            else:
                break

        if elapsed > safety * budget:
            break

    u_grid = _sample_on_uniform_grid(best_u, case_spec["output"]["grid"])

    # Optional self-check on output grid against exact solution
    if comm.rank == 0:
        grid = case_spec["output"]["grid"]
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xmin, xmax, ymin, ymax = map(float, grid["bbox"])
        xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
        ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        exact_grid = _exact_numpy(X, Y)
        grid_linf = float(np.max(np.abs(u_grid - exact_grid)))
        best_info["grid_linf_error"] = grid_linf

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best_info["mesh_resolution"]),
            "element_degree": int(best_info["element_degree"]),
            "ksp_type": str(best_info["ksp_type"]),
            "pc_type": str(best_info["pc_type"]),
            "rtol": float(best_info["rtol"]),
            "iterations": int(best_info["iterations"]),
            "l2_error": float(best_info["l2_error"]),
            "relative_l2_error": float(best_info["relative_l2_error"]),
            "grid_linf_error": float(best_info.get("grid_linf_error", np.nan)),
        },
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
