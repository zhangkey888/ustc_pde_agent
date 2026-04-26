import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _probe_function(u_func, pts):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _manufactured_expressions(domain, eps, beta):
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    grad_u = ufl.grad(u_exact)
    f_expr = -eps * ufl.div(grad_u) + ufl.dot(beta, grad_u)
    return u_exact, f_expr


def _solve_once(n, degree, use_supg):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    eps_val = 0.01
    beta_vec = np.array([10.0, 4.0], dtype=np.float64)
    eps = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))

    u_exact_ufl, f_expr = _manufactured_expressions(domain, eps, beta)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    if use_supg:
        beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
        tau = h / (2.0 * beta_norm)
        residual_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        residual_rhs = f_expr
        a += tau * residual_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * residual_rhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(l2_err),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    candidates = [
        (80, 2, True),
        (96, 2, True),
        (112, 2, True),
        (128, 2, True),
        (144, 2, True),
    ]

    target_error = 2.48e-5
    budget = 11.0
    best_u = None
    best_info = None

    for n, degree, supg in candidates:
        elapsed = time.perf_counter() - t0
        if elapsed > budget:
            break
        uh, info = _solve_once(n, degree, supg)
        best_u, best_info = uh, info
        if info["l2_error"] <= target_error and (time.perf_counter() - t0) > 0.6 * budget:
            break

    if best_u is None:
        best_u, best_info = _solve_once(80, 2, True)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_uniform_grid(best_u, grid)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": best_info["mesh_resolution"],
            "element_degree": best_info["element_degree"],
            "ksp_type": best_info["ksp_type"],
            "pc_type": best_info["pc_type"],
            "rtol": best_info["rtol"],
            "iterations": best_info["iterations"],
        },
    }

    return result


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
