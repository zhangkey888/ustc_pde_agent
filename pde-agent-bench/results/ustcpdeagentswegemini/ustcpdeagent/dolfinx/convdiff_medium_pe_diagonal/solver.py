import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_u_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _rhs_ufl(msh, eps, beta):
    x = ufl.SpatialCoordinate(msh)
    uex = _exact_u_ufl(msh)
    return -eps * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))


def _boundary_all(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
    )


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Some output grid points were not evaluated.")
        return out.reshape(ny, nx)
    return None


def _solve_once(n, degree=1):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps_val = 0.05
    beta_vec = np.array([3.0, 3.0], dtype=np.float64)
    beta = fem.Constant(msh, ScalarType(beta_vec))
    eps_c = fem.Constant(msh, ScalarType(eps_val))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact = _exact_u_ufl(msh)
    f_expr = _rhs_ufl(msh, eps_val, ufl.as_vector(beta_vec))

    h = ufl.CellDiameter(msh)
    beta_norm = float(np.linalg.norm(beta_vec))
    alpha = beta_norm * h / (2.0 * eps_val)
    tau = h / (2.0 * beta_norm) * (ufl.cosh(alpha) / ufl.sinh(alpha) - 1.0 / alpha)

    a = (
        eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * (-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _boundary_all)
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
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    u_ex_fun = fem.Function(V)
    u_ex_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_fun.x.array
    err_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    err_l2 = np.sqrt(comm.allreduce(err_sq, op=MPI.SUM))

    return {
        "mesh": msh,
        "uh": uh,
        "err_l2": float(err_l2),
        "solve_time": float(solve_time),
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(solver.getTolerances()[0]),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    wall_limit = 4.435
    target_err = 1.41e-3
    t_start = time.perf_counter()

    candidates = [48, 64, 80, 96, 112, 128]
    best = None
    for n in candidates:
        if best is not None and (time.perf_counter() - t_start) > 0.9 * wall_limit:
            break
        current = _solve_once(n, degree=1)
        best = current
        if current["err_l2"] <= target_err:
            avg = (time.perf_counter() - t_start) / (candidates.index(n) + 1)
            if (wall_limit - (time.perf_counter() - t_start)) < avg:
                break

    u_grid = _sample_on_grid(best["mesh"], best["uh"], case_spec["output"]["grid"])
    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": best["mesh_resolution"],
                "element_degree": best["element_degree"],
                "ksp_type": best["ksp_type"],
                "pc_type": best["pc_type"],
                "rtol": best["rtol"],
                "iterations": best["iterations"],
                "l2_error": best["err_l2"],
                "stabilization": "SUPG",
            },
        }
    return {"u": None, "solver_info": {}}
