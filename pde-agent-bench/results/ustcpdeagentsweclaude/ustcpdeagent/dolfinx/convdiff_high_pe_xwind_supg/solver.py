import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_on_grid(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        arr = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        arr = np.asarray(arr).reshape(len(points_on_proc), -1)[:, 0]
        vals[np.array(ids_on_proc, dtype=np.int32)] = arr

    gathered = msh.comm.allgather(vals)
    out = np.full_like(vals, np.nan)
    for g in gathered:
        mask = np.isnan(out) & ~np.isnan(g)
        out[mask] = g[mask]
    if np.isnan(out).any():
        raise RuntimeError("Some sampling points could not be evaluated.")
    return out.reshape(ny, nx)


def _manufactured_rhs(x, epsilon, beta):
    uex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -epsilon * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))
    return uex, f


def _solve_level(n, degree, epsilon, beta_vec):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    beta = ufl.as_vector((ScalarType(beta_vec[0]), ScalarType(beta_vec[1])))
    u_exact, f = _manufactured_rhs(x, epsilon, beta)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + ScalarType(1.0e-14))
    pe = beta_norm * h / (2.0 * epsilon)
    coth_pe = (ufl.exp(2.0 * pe) + 1.0) / (ufl.exp(2.0 * pe) - 1.0)
    tau_stream = h / (2.0 * beta_norm) * (coth_pe - 1.0 / pe)

    a = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    )
    L = f * v * ufl.dx

    strong_res_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    test_stream = ufl.dot(beta, ufl.grad(v))
    a += tau_stream * strong_res_u * test_stream * ufl.dx
    L += tau_stream * f * test_stream * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=1)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        ksp.solve(b, uh.x.petsc_vec)
    except Exception:
        ksp.setType("gmres")
        ksp.getPC().setType("ilu")
        ksp.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
        ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    err_sq_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))

    return msh, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": float(err_l2),
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", 0.01))
    beta_in = pde.get("beta", [20.0, 0.0])
    beta_vec = (float(beta_in[0]), float(beta_in[1]))
    grid_spec = case_spec["output"]["grid"]

    candidates = [(48, 2), (64, 2), (80, 2)]
    best = None
    for n, degree in candidates:
        best = _solve_level(n, degree, epsilon, beta_vec)

    msh, uh, info = best
    u_grid = _sample_on_grid(uh, msh, grid_spec)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
        },
    }
