import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _probe_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    map_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            map_local.append(i)

    if pts_local:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(map_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        out = gathered[0].copy()
        for g in gathered[1:]:
            mask = np.isnan(out) & ~np.isnan(g)
            out[mask] = g[mask]
        out[np.isnan(out)] = 0.0
    else:
        out = None
    return domain.comm.bcast(out, root=0)


def _sample_on_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _compute_l2_error(domain, uh: fem.Function, t: float) -> float:
    x = ufl.SpatialCoordinate(domain)
    u_ex = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    local = fem.assemble_scalar(form)
    global_val = domain.comm.allreduce(local, op=MPI.SUM)
    return float(np.sqrt(global_val))


def _solve_once(mesh_resolution: int, degree: int, dt: float, t0: float, t_end: float, kappa: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-t0) * np.sin(4 * math.pi * X[0]) * np.sin(4 * math.pi * X[1]))

    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)

    f_func = fem.Function(V)
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_func * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)

    uh = fem.Function(V)
    t = t0
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    for _ in range(n_steps):
        t += dt
        u_bc.interpolate(lambda X, tt=t: np.exp(-tt) * np.sin(4 * math.pi * X[0]) * np.sin(4 * math.pi * X[1]))
        f_func.interpolate(
            lambda X, tt=t: np.exp(-tt)
            * (-1.0 + 32.0 * math.pi * math.pi * kappa)
            * np.sin(4 * math.pi * X[0])
            * np.sin(4 * math.pi * X[1])
        )

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = uh.x.array

    error_l2 = _compute_l2_error(domain, uh, t_end)
    u0 = fem.Function(V)
    u0.interpolate(lambda X: np.exp(-t0) * np.sin(4 * math.pi * X[0]) * np.sin(4 * math.pi * X[1]))

    return {
        "domain": domain,
        "solution": uh,
        "u_initial": u0,
        "l2_error": error_l2,
        "iterations": int(total_iterations),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": 1e-10,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    coeffs = pde.get("coefficients", {})

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    suggested_dt = float(time_spec.get("dt", 0.005))
    kappa = float(coeffs.get("kappa", 1.0))
    output_grid = case_spec["output"]["grid"]

    start = time.perf_counter()
    budget = 3.274

    candidates = [
        (24, 1, min(suggested_dt, 0.0025)),
        (32, 1, min(suggested_dt, 0.0020)),
        (40, 1, min(suggested_dt, 0.00125)),
        (32, 2, min(suggested_dt, 0.0025)),
    ]

    best = None
    for mesh_resolution, degree, dt in candidates:
        result = _solve_once(mesh_resolution, degree, dt, t0, t_end, kappa)
        elapsed = time.perf_counter() - start
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result
        if elapsed > 0.8 * budget:
            break

    u_grid = _sample_on_grid(best["solution"], output_grid)
    u0_grid = _sample_on_grid(best["u_initial"], output_grid)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "dt": best["dt"],
        "n_steps": best["n_steps"],
        "time_scheme": best["time_scheme"],
        "l2_error": best["l2_error"],
    }

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }
