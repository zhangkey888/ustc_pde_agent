import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.real(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        global_vals = np.nan_to_num(global_vals, nan=0.0)
        grid = global_vals.reshape(ny, nx)
    else:
        grid = None
    return msh.comm.bcast(grid, root=0)


def _l2_norm(comm, expr):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(expr * ufl.dx)), op=MPI.SUM))


def _solve_poisson(V, rhs, bc, prefix, ksp_type, pc_type, rtol):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(rhs, v) * ufl.dx
    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1.0e-14,
        "ksp_max_it": 5000,
    }
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=prefix,
    )
    uh = problem.solve()
    its = int(problem.solver.getIterationNumber())
    return uh, its


def _solve_mixed_biharmonic(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    f_expr = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.cos(6.0 * ufl.pi * x[1])

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    w, it1 = _solve_poisson(V, f_expr, bc, f"bih1_{n}_", ksp_type, pc_type, rtol)
    u, it2 = _solve_poisson(V, w, bc, f"bih2_{n}_", ksp_type, pc_type, rtol)

    v = ufl.TestFunction(V)
    residual_form = fem.form((ufl.inner(ufl.grad(w), ufl.grad(v)) - ufl.inner(f_expr, v)) * ufl.dx)
    test_fun = fem.Function(V)
    test_fun.x.array[:] = np.random.default_rng(0).standard_normal(test_fun.x.array.size)
    test_fun.x.scatter_forward()
    weak_residual_1 = abs(fem.assemble_scalar(fem.form(
        (ufl.inner(ufl.grad(w), ufl.grad(test_fun)) - ufl.inner(f_expr, test_fun)) * ufl.dx
    )))
    weak_residual_2 = abs(fem.assemble_scalar(fem.form(
        (ufl.inner(ufl.grad(u), ufl.grad(test_fun)) - ufl.inner(w, test_fun)) * ufl.dx
    )))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(it1 + it2),
        "verification": {
            "weak_residual_poisson1": float(comm.allreduce(weak_residual_1, op=MPI.SUM)),
            "weak_residual_poisson2": float(comm.allreduce(weak_residual_2, op=MPI.SUM)),
            "u_H1_seminorm": float(_l2_norm(comm, ufl.inner(ufl.grad(u), ufl.grad(u)))),
            "w_H1_seminorm": float(_l2_norm(comm, ufl.inner(ufl.grad(w), ufl.grad(w)))),
        },
    }
    return msh, u, solver_info


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    degree = 2
    mesh_resolution = max(64, min(128, 2 * max(nx, ny)))
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    try:
        _, u, solver_info = _solve_mixed_biharmonic(
            mesh_resolution, degree, ksp_type, pc_type, rtol
        )
    except Exception:
        _, u, solver_info = _solve_mixed_biharmonic(
            max(40, mesh_resolution // 2), 1, "preonly", "lu", 1.0e-12
        )

    if MPI.COMM_WORLD.size == 1 and solver_info["mesh_resolution"] <= 80:
        try:
            _, u_ref, _ = _solve_mixed_biharmonic(
                min(128, 2 * solver_info["mesh_resolution"]),
                solver_info["element_degree"],
                "cg",
                "hypre",
                solver_info["rtol"],
            )
            g1 = _sample_on_grid(u, 21, 21, [0.0, 1.0, 0.0, 1.0])
            g2 = _sample_on_grid(u_ref, 21, 21, [0.0, 1.0, 0.0, 1.0])
            solver_info["verification"]["self_convergence_maxdiff"] = float(np.max(np.abs(g1 - g2)))
        except Exception:
            solver_info["verification"]["self_convergence_maxdiff"] = None

    u_grid = _sample_on_grid(u, nx, ny, bbox)
    return {"u": u_grid, "solver_info": solver_info}
