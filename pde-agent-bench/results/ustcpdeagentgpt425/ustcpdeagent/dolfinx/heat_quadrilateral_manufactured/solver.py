import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# DIAGNOSIS:
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
#
# METHOD:
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat


def _ufact(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _fexpr(msh, t, kappa):
    uex = _ufact(msh, t)
    return -uex - kappa * (-2.0 * ufl.pi * ufl.pi * uex)


def _interp(V, expr):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f


def _sample(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells, ids = [], [], []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals_local[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.gather(vals_local, root=0)
        if comm.rank == 0:
            merged = np.full(nx * ny, np.nan, dtype=np.float64)
            for arr in gathered:
                mask = ~np.isnan(arr)
                merged[mask] = arr[mask]
        else:
            merged = None
        merged = comm.bcast(merged, root=0)
    else:
        merged = vals_local

    return merged.reshape((ny, nx))


def _solve_once(mesh_resolution, degree, dt, t0, t_end, kappa):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, ScalarType(dt))
    kappa_c = fem.Constant(msh, ScalarType(kappa))
    t_c = fem.Constant(msh, ScalarType(t0))

    u_n = _interp(V, _ufact(msh, t_c))
    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array
    uh.x.scatter_forward()

    f_fun = _interp(V, _fexpr(msh, t_c, kappa_c))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = _interp(V, _ufact(msh, t_c))
    bc = fem.dirichletbc(u_bc, bdofs)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(fem.Expression(_ufact(msh, t_c), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_fexpr(msh, t_c, kappa_c), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            total_iterations += max(solver.getIterationNumber(), 0)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start

    u_ex = _interp(V, _ufact(msh, ScalarType(t_end)))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))

    return msh, uh, l2_err, wall, total_iterations, n_steps, dt


def solve(case_spec: dict) -> dict:
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))
    kappa = float(case_spec.get("coefficients", {}).get("kappa", 1.0))
    grid = case_spec["output"]["grid"]

    candidates = [
        {"mesh_resolution": 28, "degree": 1, "dt": min(dt_suggested, 0.01)},
        {"mesh_resolution": 32, "degree": 2, "dt": min(dt_suggested, 0.01)},
        {"mesh_resolution": 40, "degree": 2, "dt": 0.005},
        {"mesh_resolution": 48, "degree": 2, "dt": 0.004},
        {"mesh_resolution": 56, "degree": 2, "dt": 0.0025},
    ]

    target_error = 1.11e-3
    budget = 18.486
    chosen = None

    for cfg in candidates:
        out = _solve_once(cfg["mesh_resolution"], cfg["degree"], cfg["dt"], t0, t_end, kappa)
        msh, uh, l2_err, wall, iterations, n_steps, dt_used = out
        chosen = (cfg, msh, uh, l2_err, wall, iterations, n_steps, dt_used)
        if l2_err <= target_error:
            idx = candidates.index(cfg)
            if wall < 0.35 * budget and idx + 1 < len(candidates):
                continue
            break

    cfg, msh, uh, l2_err, wall, iterations, n_steps, dt_used = chosen
    u_grid = _sample(msh, uh, grid)

    V0 = fem.functionspace(msh, ("Lagrange", cfg["degree"]))
    u0 = _interp(V0, _ufact(msh, ScalarType(t0)))
    u0_grid = _sample(msh, u0, grid)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(cfg["mesh_resolution"]),
            "element_degree": int(cfg["degree"]),
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(iterations),
            "dt": float(dt_used),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "verification_l2_error": float(l2_err),
            "measured_wall_time": float(wall),
        },
    }
