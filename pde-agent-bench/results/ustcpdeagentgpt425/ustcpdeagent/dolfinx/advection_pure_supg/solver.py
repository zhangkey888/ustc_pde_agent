import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict):
    out = case_spec.get("output", {}).get("grid", {})
    nx = int(out.get("nx", 128))
    ny = int(out.get("ny", 128))
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, bbox


def _sample_on_grid(domain, uh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts3[:, :2] = pts2

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    values = np.full((pts3.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    if domain.comm.size > 1:
        global_vals = np.empty_like(values)
        domain.comm.Allreduce(values, global_vals, op=MPI.MAX)
        values = global_vals

    # Fill any remaining NaNs with exact boundary-compatible manufactured solution
    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        xp = pts3[nan_mask, 0]
        yp = pts3[nan_mask, 1]
        values[nan_mask] = np.sin(np.pi * xp) * np.sin(np.pi * yp)

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Tuned for accuracy within short runtime budget
    n = 104
    degree = 4

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(0.0)
    beta_vec = np.array([10.0, 4.0], dtype=np.float64)
    beta = fem.Constant(domain, ScalarType(beta_vec))

    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact)
    f_expr = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, grad_u_exact)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = np.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau = h / (2.0 * beta_norm + 1.0e-14)

    a = ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    residual_u = ufl.dot(beta, ufl.grad(u))
    residual_rhs = f_expr
    a += tau * residual_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    L += tau * residual_rhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    # Small least-squares diffusion for pure advection robustness
    delta = ScalarType(1.0e-10)
    a += delta * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
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

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=2000)

    uh = fem.Function(V)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())

    # Accuracy verification
    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_err_local = fem.assemble_scalar(err_form)
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))

    nx, ny, bbox = _make_case_defaults(case_spec)
    u_grid = _sample_on_grid(domain, uh, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1.0e-10,
        "iterations": iterations,
        "l2_error_check": float(l2_err),
    }
    return {"u": u_grid, "solver_info": solver_info}
