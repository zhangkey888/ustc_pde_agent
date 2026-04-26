import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _probe_points(u_func: fem.Function, points_array: np.ndarray) -> np.ndarray:
    """
    Evaluate scalar FEM function at points_array of shape (3, N).
    Returns array of shape (N,).
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    ptsT = points_array.T
    cell_candidates = geometry.compute_collisions_points(bb_tree, ptsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, ptsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(ptsT[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )
    vals = _probe_points(u_func, points)
    return vals.reshape(ny, nx)


def _manufactured_exact_expr(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _solve_once(mesh_resolution: int, degree: int, kappa_value: float,
                ksp_type: str, pc_type: str, rtol: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                  cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = fem.Constant(msh, ScalarType(kappa_value))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
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
    solver.setFromOptions()

    uh = fem.Function(V)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    local_l2 = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    global_l2 = comm.allreduce(local_l2, op=MPI.SUM)
    l2_error = float(np.sqrt(global_l2))

    its = int(solver.getIterationNumber())
    return msh, uh, l2_error, its


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.perf_counter()

    # Problem data from task
    kappa_value = 0.1

    # Adaptive accuracy/time trade-off: try increasingly accurate settings
    # while respecting the tight wall-time budget.
    candidates = [
        (24, 1, "cg", "hypre", 1e-10),
        (32, 1, "cg", "hypre", 1e-10),
        (40, 1, "cg", "hypre", 1e-10),
        (24, 2, "cg", "hypre", 1e-10),
        (32, 2, "cg", "hypre", 1e-10),
        (40, 2, "cg", "hypre", 1e-10),
    ]

    best = None
    target_time = 1.245 * 0.9  # keep some safety margin

    for mesh_resolution, degree, ksp_type, pc_type, rtol in candidates:
        elapsed_before = time.perf_counter() - t0
        if elapsed_before > target_time and best is not None:
            break
        try:
            msh, uh, l2_error, iterations = _solve_once(
                mesh_resolution, degree, kappa_value, ksp_type, pc_type, rtol
            )
            elapsed_after = time.perf_counter() - t0
            best = {
                "mesh": msh,
                "u": uh,
                "l2_error": l2_error,
                "iterations": iterations,
                "mesh_resolution": mesh_resolution,
                "element_degree": degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "elapsed": elapsed_after,
            }
            if elapsed_after > target_time:
                break
        except Exception:
            continue

    if best is None:
        # Fallback robust direct solve
        msh, uh, l2_error, iterations = _solve_once(
            24, 1, kappa_value, "preonly", "lu", 1e-10
        )
        best = {
            "mesh": msh,
            "u": uh,
            "l2_error": l2_error,
            "iterations": iterations,
            "mesh_resolution": 24,
            "element_degree": 1,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "elapsed": time.perf_counter() - t0,
        }

    u_grid = _sample_on_uniform_grid(best["u"], case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error": float(best["l2_error"]),
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
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
