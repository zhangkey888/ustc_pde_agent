import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x, y):
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_on_proc = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_on_proc.append(i)

    if points_on_proc:
        ev = uh.eval(np.array(points_on_proc, dtype=np.float64),
                     np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(idx_on_proc, dtype=np.int32)] = np.asarray(ev, dtype=np.float64).reshape(-1)

    if domain.comm.size > 1:
        recv = np.empty_like(vals)
        domain.comm.Allreduce(vals, recv, op=MPI.MAX)
        vals = recv

    return vals.reshape(ny, nx)


def _solve_once(nx, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_ex = ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(8 * pi * x[0]) * ufl.sin(8 * pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_ex))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solver failed")
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()

    u_exact_h = fem.Function(V)
    u_exact_h.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    err_form = fem.form((uh - u_exact_h) * (uh - u_exact_h) * ufl.dx)
    l2_sq_local = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_err = math.sqrt(max(l2_sq, 0.0))

    return domain, uh, l2_err, int(solver.getIterationNumber()), ksp_type, pc_type


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    time_limit = 1.942
    candidates = [(28, 1), (36, 1), (24, 2), (32, 2), (40, 2)]

    best = None
    chosen = None

    for nx, degree in candidates:
        domain, uh, l2_err, iters, ksp_type, pc_type = _solve_once(nx, degree)
        best = (nx, degree, domain, uh, l2_err, iters, ksp_type, pc_type)
        if l2_err <= 3.55e-03:
            chosen = best
        if time.perf_counter() - t0 > 0.9 * time_limit:
            break

    if chosen is None:
        chosen = best

    nx, degree, domain, uh, l2_err, iters, ksp_type, pc_type = chosen
    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(1e-10),
        "iterations": int(iters),
        "verification_l2_fem": float(l2_err),
        "wall_time_estimate": float(time.perf_counter() - t0),
    }
    return {"u": u_grid, "solver_info": solver_info}
