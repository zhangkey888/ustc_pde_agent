import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_np(x, y):
    return np.sin(2.0 * np.pi * (x + y)) * np.sin(np.pi * (x - y))


def _build_problem(n, degree, ksp_type="gmres", pc_type="ilu", rtol=1.0e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    eps = fem.Constant(msh, ScalarType(0.05))
    beta_arr = np.array([3.0, 1.0], dtype=np.float64)
    beta = fem.Constant(msh, beta_arr.astype(ScalarType))
    beta_norm = float(np.linalg.norm(beta_arr))

    u_exact = ufl.sin(2.0 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    f = -eps * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx

    h = ufl.CellDiameter(msh)
    pe = beta_norm * h / (2.0 * eps)
    tau = h / (2.0 * beta_norm) * ((ufl.cosh(pe) / ufl.sinh(pe)) - 1.0 / pe)
    r_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a += tau * ufl.dot(beta, ufl.grad(v)) * r_u * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2.0 * np.pi * (X[0] + X[1])) * np.sin(np.pi * (X[0] - X[1])))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-12, max_it=2000)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("iterative KSP failed")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=1.0e-12, max_it=1)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    return uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(l2_error),
    }


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    pts_local = []
    cells_local = []
    point_ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            point_ids.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(point_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_uniform_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    grid = case_spec["output"]["grid"]
    time_limit = float(case_spec.get("time_limit", 2.402))
    budget = min(time_limit, 2.35)

    candidates = [(56, 1), (72, 1), (96, 1), (128, 1), (80, 2)]
    chosen = None

    for n, degree in candidates:
        if chosen is not None and (time.perf_counter() - start) > 0.82 * budget:
            break
        uh, info = _build_problem(n, degree)
        chosen = (uh, info)
        if info["l2_error"] <= 5.10e-03 and (time.perf_counter() - start) > 0.72 * budget:
            break

    if chosen is None:
        uh, info = _build_problem(56, 1)
    else:
        uh, info = chosen

    u_grid = _sample_on_uniform_grid(uh, grid)
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
