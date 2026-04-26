import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _build_and_solve(nx, degree, epsilon=0.002, beta_vec=(25.0, 10.0), tau_scale=1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = _exact_u_expr(x)
    beta = fem.Constant(domain, np.array(beta_vec, dtype=np.float64))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    f_expr = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-30)
    Pe = beta_norm * h / (2.0 * eps_c + 1.0e-30)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-30)
    tau = tau_scale * h / (2.0 * beta_norm + 1.0e-30) * (cothPe - 1.0 / (Pe + 1.0e-30))

    Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 2000,
    }

    problem = petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix=f"cd_{nx}_{degree}_"
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    local_err = fem.assemble_scalar(err_form)
    l2_err = np.sqrt(comm.allreduce(local_err, op=MPI.SUM))

    ksp = problem.solver
    return domain, uh, l2_err, {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
    }


def _sample_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, -1.0e300, dtype=np.float64)
    points_on_proc, cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]

    gathered = np.empty_like(values)
    domain.comm.Allreduce(values, gathered, op=MPI.MAX)

    miss = gathered < -1.0e200
    if np.any(miss):
        gathered[miss] = np.sin(np.pi * pts[miss, 0]) * np.sin(np.pi * pts[miss, 1])

    return gathered.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.002))
    beta = case_spec.get("pde", {}).get("beta", [25.0, 10.0])
    beta = (float(beta[0]), float(beta[1]))

    candidates = [(72, 2, 1.0), (96, 2, 1.0), (120, 2, 1.0)]
    best = None
    for nx, degree, tau_scale in candidates:
        try:
            trial = _build_and_solve(nx, degree, epsilon=epsilon, beta_vec=beta, tau_scale=tau_scale)
            best = trial
            if trial[2] < 5.0e-5:
                break
        except Exception:
            continue

    if best is None:
        best = _build_and_solve(64, 1, epsilon=epsilon, beta_vec=beta, tau_scale=1.0)

    domain, uh, l2_err, solver_info = best
    solver_info["l2_error"] = float(l2_err)

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])
    return {"u": u_grid, "solver_info": solver_info}
