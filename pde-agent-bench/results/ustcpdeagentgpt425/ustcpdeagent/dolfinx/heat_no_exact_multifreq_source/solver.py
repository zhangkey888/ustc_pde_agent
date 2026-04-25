import time
import math
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
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType


def _get_time_params(case_spec):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.12))
    dt = float(pde.get("dt", 0.02))
    if dt <= 0:
        dt = 0.02
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    return t0, t_end, dt, n_steps


def _source_expression(x):
    return np.sin(5.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1]) + 0.5 * np.sin(
        9.0 * np.pi * x[0]
    ) * np.sin(7.0 * np.pi * x[1])


def _exact_solution(x, t, kappa=1.0):
    lam1 = np.pi**2 * (5.0**2 + 3.0**2) * kappa
    lam2 = np.pi**2 * (9.0**2 + 7.0**2) * kappa
    a1 = (1.0 - np.exp(-lam1 * t)) / lam1
    a2 = 0.5 * (1.0 - np.exp(-lam2 * t)) / lam2
    return a1 * np.sin(5.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1]) + a2 * np.sin(
        9.0 * np.pi * x[0]
    ) * np.sin(7.0 * np.pi * x[1])


def _probe_function(u_func, pts):
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)
    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)
    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _run_solver(mesh_resolution, degree, dt, n_steps, kappa=1.0):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    uh = fem.Function(V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1]) + 0.5 * ufl.sin(
        9.0 * ufl.pi * x[0]
    ) * ufl.sin(7.0 * ufl.pi * x[1])
    f = fem.Expression(f_expr, V.element.interpolation_points)
    f_fun = fem.Function(V)
    f_fun.interpolate(f)

    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

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
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    total_iterations = 0
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        if its is not None:
            total_iterations += int(its)
        u_n.x.array[:] = uh.x.array

    err_form = fem.form((uh - _interpolated_exact(V, n_steps * dt, kappa)) ** 2 * ufl.dx)
    l2_sq_local = fem.assemble_scalar(err_form)
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_err = math.sqrt(max(l2_sq, 0.0))
    return domain, uh, total_iterations, l2_err, solver


def _interpolated_exact(V, t, kappa):
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda x: _exact_solution(x, t, kappa))
    return u_ex


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0, t_end, dt_base, n_steps_base = _get_time_params(case_spec)
    grid_spec = case_spec["output"]["grid"]

    start = time.perf_counter()

    candidates = [(96, 2, dt_base / 6.0), (128, 2, dt_base / 8.0), (160, 2, dt_base / 10.0)]
    best = None
    target_time = 24.0

    for mesh_resolution, degree, dt_try in candidates:
        n_steps = max(1, int(round((t_end - t0) / dt_try)))
        dt_used = (t_end - t0) / n_steps
        try:
            run_start = time.perf_counter()
            domain, uh, total_iterations, l2_err, solver = _run_solver(mesh_resolution, degree, dt_used, n_steps)
            run_time = time.perf_counter() - run_start
            best = {
                "domain": domain,
                "uh": uh,
                "iterations": total_iterations,
                "l2_err": l2_err,
                "mesh_resolution": mesh_resolution,
                "degree": degree,
                "dt": dt_used,
                "n_steps": n_steps,
                "ksp_type": solver.getType(),
                "pc_type": solver.getPC().getType(),
                "rtol": solver.getTolerances()[0],
                "runtime": run_time,
            }
            if run_time > target_time or l2_err < 5e-3:
                break
        except Exception:
            continue

    if best is None:
        mesh_resolution, degree = 40, 1
        n_steps = max(1, int(round((t_end - t0) / dt_base)))
        dt_used = (t_end - t0) / n_steps
        domain, uh, total_iterations, l2_err, solver = _run_solver(mesh_resolution, degree, dt_used, n_steps)
        best = {
            "domain": domain,
            "uh": uh,
            "iterations": total_iterations,
            "l2_err": l2_err,
            "mesh_resolution": mesh_resolution,
            "degree": degree,
            "dt": dt_used,
            "n_steps": n_steps,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": solver.getTolerances()[0],
            "runtime": time.perf_counter() - start,
        }

    u_grid = _sample_on_grid(best["uh"], grid_spec)
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    u_initial = np.zeros((ny, nx), dtype=np.float64)

    result = {
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
            "verification_l2_error_vs_exact": float(best["l2_err"]),
            "wall_time_sec": float(time.perf_counter() - start),
        },
    }
    return result
