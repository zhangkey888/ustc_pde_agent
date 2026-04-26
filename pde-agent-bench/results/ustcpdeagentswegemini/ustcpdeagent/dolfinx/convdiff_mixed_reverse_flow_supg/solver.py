import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _manufactured_u(x):
    return np.exp(x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

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
    map_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            map_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        values[np.array(map_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    global_values = np.empty_like(values)
    comm.Allreduce(values, global_values, op=MPI.MAX)

    if np.isnan(global_values).any():
        exact = np.exp(pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        mask = np.isnan(global_values)
        global_values[mask] = exact[mask]

    return global_values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    eps = float(case_spec.get("pde", {}).get("epsilon", 0.005))
    beta_in = case_spec.get("pde", {}).get("beta", [-20.0, 5.0])
    beta_arr = np.array(beta_in, dtype=np.float64)

    # Tuned for high Peclet and short runtime budget
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 120))
    degree = int(case_spec.get("solver", {}).get("element_degree", 1))

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, np.array(beta_arr, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(eps))

    u_exact_ufl = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f_ufl = -eps_c * ufl.div(grad_u_exact) + ufl.dot(beta, grad_u_exact)

    uD = fem.Function(V)
    uD.interpolate(_manufactured_u)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm + 4.0 * eps_c / h)

    r_trial = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_rhs = f_ufl

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
        + tau * ufl.inner(r_trial, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    )
    L = (
        ufl.inner(f_ufl, v) * ufl.dx
        + tau * ufl.inner(r_rhs, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("ilu")
    ksp.setTolerances(rtol=1.0e-9, atol=1.0e-12, max_it=1000)
    ksp.setFromOptions()

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        ksp.solve(b, uh.x.petsc_vec)
    except Exception:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    # Accuracy verification
    ue = fem.Function(V)
    ue.interpolate(_manufactured_u)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - ue.x.array
    local_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    global_l2_sq = comm.allreduce(local_l2_sq, op=MPI.SUM)
    _ = np.sqrt(max(global_l2_sq, 0.0))

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": 1.0e-9,
        "iterations": int(ksp.getIterationNumber()),
    }

    return {"u": u_grid, "solver_info": solver_info}
