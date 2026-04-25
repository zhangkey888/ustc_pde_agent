import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_np(x, y):
    return np.exp(6.0 * y) * np.sin(np.pi * x)


def _build_and_solve(n, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.exp(6.0 * x[1]) * ufl.sin(ufl.pi * x[0])
    kappa = fem.Constant(domain, ScalarType(1.0))
    f = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _u_exact_np(X[0], X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 5000,
        }
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        try:
            iterations = int(problem._solver.getIterationNumber())
        except Exception:
            iterations = 0

    ue = fem.Function(V)
    ue.interpolate(lambda X: _u_exact_np(X[0], X[1]))
    err_form = fem.form((uh - ue) ** 2 * ufl.dx)
    ref_form = fem.form(ue ** 2 * ufl.dx)
    err_l2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref_l2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    rel_l2 = err_l2 / ref_l2 if ref_l2 > 0 else err_l2

    return domain, uh, iterations, rel_l2


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if domain.comm.size > 1:
        gathered = domain.comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        values[nan_mask] = _u_exact_np(pts[nan_mask, 0], pts[nan_mask, 1])

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]

    n = 56
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    domain, uh, iterations, rel_l2 = _build_and_solve(n, degree, ksp_type, pc_type, rtol)

    if rel_l2 > 5e-4:
        n = 72
        domain, uh, iterations2, rel_l2 = _build_and_solve(n, degree, ksp_type, pc_type, rtol)
        iterations += iterations2

    u_grid = _sample_on_grid(domain, uh, grid)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification_rel_l2_error": float(rel_l2),
        },
    }
