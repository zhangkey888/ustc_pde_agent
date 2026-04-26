import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)


def _build_exact_ufl(x, t):
    return ufl.exp(-t) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])


def _build_rhs_ufl(x, t, kappa):
    u_ex = _build_exact_ufl(x, t)
    return -u_ex + kappa * 32.0 * (ufl.pi ** 2) * u_ex


def _interpolate_exact(fun, t):
    fun.interpolate(
        lambda X: np.exp(-t)
        * np.sin(4.0 * np.pi * X[0])
        * np.sin(4.0 * np.pi * X[1])
    )


def _sample_on_grid(domain, u_fun, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64),
                          np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            m = ~np.isnan(arr)
            merged[m] = arr[m]
        if np.isnan(merged).any():
            raise RuntimeError("Point evaluation failed for some grid points")
        return merged.reshape((ny, nx))
    return None


def _compute_l2_error(domain, uh, t):
    x = ufl.SpatialCoordinate(domain)
    u_ex = _build_exact_ufl(x, ScalarType(t))
    form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    local = fem.assemble_scalar(form)
    global_sq = domain.comm.allreduce(local, op=MPI.SUM)
    return float(np.sqrt(global_sq))


def _run(mesh_resolution, degree, dt, t_end, kappa, rtol):
    domain = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_n = fem.Function(V)
    uh = fem.Function(V)
    u_bc = fem.Function(V)
    _interpolate_exact(u_n, 0.0)
    _interpolate_exact(u_bc, 0.0)
    uh.x.array[:] = u_n.x.array
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    t_var = fem.Constant(domain, ScalarType(dt))
    f_ufl = _build_rhs_ufl(x, t_var, ScalarType(kappa))
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    n_steps = int(round(t_end / dt))
    t = 0.0
    total_iterations = 0

    for _ in range(n_steps):
        t += dt
        t_var.value = ScalarType(t)
        _interpolate_exact(u_bc, t)
        f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            ksp.solve(b, uh.x.petsc_vec)
        except Exception:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setOperators(A)
            ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += max(int(ksp.getIterationNumber()), 0)
        u_n.x.array[:] = uh.x.array

    return {
        "domain": domain,
        "u": uh,
        "l2_error": _compute_l2_error(domain, uh, t_end),
        "mesh_resolution": mesh_resolution,
        "degree": degree,
        "dt": dt,
        "n_steps": n_steps,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": rtol,
        "iterations": total_iterations,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = case_spec.get("time", {})
    coeffs = case_spec.get("coefficients", {})
    grid = case_spec["output"]["grid"]

    kappa = float(coeffs.get("kappa", 1.0))
    t0 = float(time_spec.get("t0", pde.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", pde.get("t_end", 0.06)))
    dt_suggested = float(time_spec.get("dt", pde.get("dt", 0.003)))
    final_time = t_end - t0

    wall_budget = 35.095
    start = time.perf_counter()

    candidates = [
        (40, 1, dt_suggested),
        (56, 1, dt_suggested),
        (64, 2, dt_suggested),
        (80, 2, dt_suggested),
        (80, 2, dt_suggested / 2.0),
        (96, 2, dt_suggested / 2.0),
    ]

    best = None
    for mesh_resolution, degree, dt in candidates:
        if best is not None and (time.perf_counter() - start) > 0.8 * wall_budget:
            break
        result = _run(mesh_resolution, degree, dt, final_time, kappa, 1e-10)
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result
        if result["l2_error"] <= 2.43e-2 and (time.perf_counter() - start) > 0.4 * wall_budget:
            break

    if best is None:
        raise RuntimeError("No successful solve completed")

    u_grid = _sample_on_grid(best["domain"], best["u"], grid)

    if COMM.rank == 0:
        nx = int(grid["nx"])
        ny = int(grid["ny"])
        xmin, xmax, ymin, ymax = map(float, grid["bbox"])
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_initial = _exact_numpy(XX, YY, 0.0).reshape((ny, nx))
        return {
            "u": u_grid,
            "u_initial": u_initial,
            "solver_info": {
                "mesh_resolution": int(best["mesh_resolution"]),
                "element_degree": int(best["degree"]),
                "ksp_type": str(best["ksp_type"]),
                "pc_type": str(best["pc_type"]),
                "rtol": float(best["rtol"]),
                "iterations": int(best["iterations"]),
                "dt": float(best["dt"]),
                "n_steps": int(best["n_steps"]),
                "time_scheme": "backward_euler",
                "l2_error": float(best["l2_error"]),
                "wall_time_sec": float(time.perf_counter() - start),
            },
        }

    return {"u": None, "u_initial": None, "solver_info": None}
