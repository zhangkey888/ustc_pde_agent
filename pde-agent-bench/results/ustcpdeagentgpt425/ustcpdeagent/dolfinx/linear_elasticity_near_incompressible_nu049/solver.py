import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    E = 1.0
    nu = 0.49
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    time_limit = 4.217
    target_err = 2.89e-5
    degree = 2
    mesh_candidates = [24, 32, 40, 48, 56, 64]

    best = None
    t0 = time.perf_counter()

    for n in mesh_candidates:
        if time.perf_counter() - t0 > 0.9 * time_limit:
            break
        try:
            candidate = _solve_once(n, degree, mu, lam, E, nu, nx_out, ny_out, (xmin, xmax, ymin, ymax))
            best = candidate
            if candidate["solver_info"]["l2_error"] <= target_err and time.perf_counter() - t0 > 0.55 * time_limit:
                break
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Failed to compute elasticity solution.")

    return best


def _solve_once(n, degree, mu, lam, E, nu, nx_out, ny_out, bbox):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(msh)

    u_exact = ufl.as_vector([
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    ])

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    ksp_type = "cg"
    pc_type = "gamg"
    rtol = 1e-10
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "cg",
                "ksp_rtol": rtol,
                "pc_type": "gamg",
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_pc_type": "jacobi",
            },
            petsc_options_prefix=f"elas_{n}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())
        if int(problem.solver.getConvergedReason()) <= 0:
            raise RuntimeError("CG/GAMG did not converge")
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        rtol = 0.0
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            petsc_options_prefix=f"elaslu_{n}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())

    u_ex_fun = fem.Function(V)
    u_ex_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_ex_fun.x.array
    err.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))
    max_error = comm.allreduce(np.max(np.abs(err.x.array)) if err.x.array.size else 0.0, op=MPI.MAX)

    u_grid = _sample_magnitude(uh, nx_out, ny_out, bbox)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error": float(l2_error),
            "max_error": float(max_error),
            "E": float(E),
            "nu": float(nu),
        },
    }


def _sample_magnitude(u_fun, nx, ny, bbox):
    msh = u_fun.function_space.mesh
    comm = msh.comm
    xmin, xmax, ymin, ymax = bbox

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    point_list = []
    cell_list = []
    idx_list = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            point_list.append(pts[i])
            cell_list.append(links[0])
            idx_list.append(i)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    if point_list:
        vals = u_fun.eval(np.array(point_list, dtype=np.float64), np.array(cell_list, dtype=np.int32))
        local_vals[np.array(idx_list, dtype=np.int32)] = vals

    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr[:, 0])
            vals[mask] = arr[mask]
        mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)
    else:
        mag = None

    return comm.bcast(mag, root=0)
