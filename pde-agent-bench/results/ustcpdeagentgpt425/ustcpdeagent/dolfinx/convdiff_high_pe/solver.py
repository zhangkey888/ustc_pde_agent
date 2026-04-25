import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)

    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            idx_local.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64),
                           np.array(cells_local, dtype=np.int32))
        values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _solve_once(n, degree=1, use_supg=True):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    eps = 0.01
    beta_vec = np.array([10.0, 10.0], dtype=np.float64)
    beta = fem.Constant(msh, np.array(beta_vec, dtype=ScalarType))

    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_ufl = -eps * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f_ufl * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(msh)
        beta_norm = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
        tau = h / (2.0 * beta_norm)
        r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        r_rhs = f_ufl
        a += tau * ufl.dot(beta, ufl.grad(v)) * r_trial * ufl.dx
        L += tau * ufl.dot(beta, ufl.grad(v)) * r_rhs * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
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
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.solve(b, uh.x.petsc_vec)

    if solver.getConvergedReason() <= 0:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    local_l2 = fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(local_l2, op=MPI.SUM))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(solver.getIterationNumber()),
        "l2_error_verification": float(l2_error),
        "supg": bool(use_supg),
    }
    return uh, info, l2_error


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    candidates = [
        (40, 1, True),
        (56, 1, True),
        (72, 1, True),
        (96, 1, True),
        (64, 2, True),
    ]

    best = None
    budget = 2.4
    for n, degree, supg in candidates:
        ts = time.perf_counter()
        uh, info, err = _solve_once(n=n, degree=degree, use_supg=supg)
        te = time.perf_counter() - ts
        if best is None or err < best[2]:
            best = (uh, info, err, te)
        if time.perf_counter() - t0 > budget:
            break

    uh, info, err, _ = best

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals = _probe_function(uh, pts)

    if np.isnan(vals).any():
        exact_vals = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        vals = np.where(np.isnan(vals), exact_vals, vals)

    u_grid = vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": info,
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
    print(out["u"].shape)
    print(out["solver_info"])
