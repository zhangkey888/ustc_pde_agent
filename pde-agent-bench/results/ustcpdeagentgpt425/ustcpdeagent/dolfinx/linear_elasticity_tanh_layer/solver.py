import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: linear_elasticity
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: vector
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: boundary_layer
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: linear_elasticity
# ```

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    pde = case_spec.get("pde", {})
    E = float(pde.get("E", 1.0))
    nu = float(pde.get("nu", 0.3))

    # Adaptive spatial accuracy within time budget
    start = time.perf_counter()
    budget = 6.974
    safety_fraction = 0.8
    candidates = [40, 56, 72, 88, 104, 120]
    degree = 2 if nu > 0.4 else 2

    best = None
    last_runtime = None
    for n in candidates:
        elapsed = time.perf_counter() - start
        if elapsed > safety_fraction * budget:
            break
        if best is not None and last_runtime is not None and elapsed + 1.5 * last_runtime > safety_fraction * budget:
            break
        try:
            result = _solve_once(E, nu, n, degree, nx, ny, bbox, force_direct=False)
        except Exception:
            result = _solve_once(E, nu, n, degree, nx, ny, bbox, force_direct=True)
        best = result
        last_runtime = result["runtime"]

    if best is None:
        best = _solve_once(E, nu, 40, degree, nx, ny, bbox, force_direct=True)

    return {"u": best["u_grid"], "solver_info": best["solver_info"]}


def _solve_once(E, nu, n, degree, nx_out, ny_out, bbox, force_direct=False):
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(pi * x[0]),
        0.1 * ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1]),
    ])

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(gdim)

    f = -ufl.div(sigma(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    ksp_type = "preonly" if force_direct else "cg"
    pc_type = "lu" if force_direct else "hypre"
    rtol = 1e-10

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if not force_direct:
        opts["pc_hypre_type"] = "boomeramg"

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=f"linear_elasticity_{n}_{degree}_{'lu' if force_direct else 'amg'}_",
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 1 if force_direct else 0
    try:
        solver = problem.solver
        if solver is not None:
            iterations = int(solver.getIterationNumber())
            ksp_type = solver.getType() or ksp_type
            pc = solver.getPC()
            if pc is not None:
                pc_type = pc.getType() or pc_type
    except Exception:
        pass

    u_grid = _sample_magnitude(uh, domain, nx_out, ny_out, bbox)
    exact_grid = _exact_magnitude(nx_out, ny_out, bbox)
    grid_err = float(np.sqrt(np.mean((u_grid - exact_grid) ** 2)))
    runtime = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification_grid_l2_error": grid_err,
        "wall_time_sec": runtime,
    }
    return {"u_grid": u_grid, "solver_info": solver_info, "runtime": runtime}


def _exact_magnitude(nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    u1 = np.tanh(6.0 * (Y - 0.5)) * np.sin(np.pi * X)
    u2 = 0.1 * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)
    return np.sqrt(u1 * u1 + u2 * u2)


def _sample_magnitude(u_sol, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _eval_points(u_sol, domain, pts)
    mag = np.linalg.norm(vals[:, :domain.geometry.dim], axis=1)
    return mag.reshape(ny, nx)


def _eval_points(u_func, domain, points):
    comm = domain.comm
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    local_values = np.full((points.shape[0], u_func.function_space.element.value_size), np.nan, dtype=np.float64)
    p_local = []
    c_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            p_local.append(points[i])
            c_local.append(links[0])
            idx_local.append(i)

    if p_local:
        vals = u_func.eval(np.array(p_local, dtype=np.float64), np.array(c_local, dtype=np.int32))
        local_values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals, dtype=np.float64)

    gathered = comm.allgather(local_values)
    values = np.full_like(local_values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr[:, 0])
        values[mask] = arr[mask]

    nan_mask = np.isnan(values[:, 0])
    if np.any(nan_mask):
        repair = points[nan_mask].copy()
        repair[:, 0] = np.clip(repair[:, 0], 0.0, 1.0 - 1e-12)
        repair[:, 1] = np.clip(repair[:, 1], 0.0, 1.0 - 1e-12)
        candidates = geometry.compute_collisions_points(tree, repair)
        colliding = geometry.compute_colliding_cells(domain, candidates, repair)
        rp, rc, ridx = [], [], []
        base_idx = np.where(nan_mask)[0]
        for j in range(repair.shape[0]):
            links = colliding.links(j)
            if len(links) > 0:
                rp.append(repair[j])
                rc.append(links[0])
                ridx.append(base_idx[j])
        if rp:
            vals = u_func.eval(np.array(rp, dtype=np.float64), np.array(rc, dtype=np.int32))
            values[np.array(ridx, dtype=np.int32)] = np.asarray(vals, dtype=np.float64)

    values[np.isnan(values)] = 0.0
    return values
