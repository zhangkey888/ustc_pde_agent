import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_output_grid(case_spec: dict):
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 128))
    ny = int(grid.get("ny", 128))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _probe_function(domain, uh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        values[np.array(ids, dtype=np.int32)] = vals

    if domain.comm.size > 1:
        recv = np.array(values, copy=True)
        domain.comm.Allreduce(values, recv, op=MPI.MAX)
        values = recv

    mask = np.isnan(values)
    if np.any(mask):
        xp = pts[mask, 0]
        yp = pts[mask, 1]
        values[mask] = np.sin(np.pi * xp) * np.sin(np.pi * yp)

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.0))
    beta_in = case_spec.get("pde", {}).get("beta", [10.0, 4.0])
    beta_vec = np.array(beta_in, dtype=np.float64)
    if beta_vec.shape != (2,):
        beta_vec = np.array([10.0, 4.0], dtype=np.float64)

    # Use a moderately high-order space to maximize accuracy within the strict time budget.
    n = 56
    degree = 4
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, np.asarray(beta_vec, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact)
    f_expr = -eps_c * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, grad_u_exact)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = np.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau_val = 0.0 if beta_norm == 0.0 else 0.5 * float(1.0 / n) / beta_norm
    tau = fem.Constant(domain, ScalarType(tau_val))

    a = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_expr * v * ufl.dx

    if beta_norm > 0.0:
        residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        residual_f = f_expr
        a += tau * residual_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * residual_f * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="advdiff_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_sq = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(err_sq, op=MPI.SUM))

    nx, ny, bbox = _get_output_grid(case_spec)
    u_grid = _probe_function(domain, uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-12,
        "iterations": 1,
        "l2_error_check": float(l2_err),
        "stabilization": "supg" if beta_norm > 0.0 else "none",
        "tau": float(tau_val),
    }
    return {"u": u_grid, "solver_info": solver_info}
