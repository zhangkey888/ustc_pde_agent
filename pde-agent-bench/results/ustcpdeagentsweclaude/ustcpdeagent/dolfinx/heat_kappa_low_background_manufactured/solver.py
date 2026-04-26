import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fp
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
# special_notes: manufactured_solution, variable_coeff
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


def _build_safe_namespace(x0, x1):
    return {
        "__builtins__": {},
        "x": x0,
        "y": x1,
        "z": 0 * x0,
        "pi": ufl.pi,
        "exp": ufl.exp,
        "sin": ufl.sin,
        "cos": ufl.cos,
        "sqrt": ufl.sqrt,
    }


def _kappa_from_spec(expr: str, x):
    return eval(expr, _build_safe_namespace(x[0], x[1]))


def _u_exact_expr(x, t):
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _f_exact_expr(x, t, kappa_expr):
    uex = _u_exact_expr(x, t)
    return -uex - ufl.div(kappa_expr * ufl.grad(uex))


def _probe_function(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tdim = msh.topology.dim
    tree = geometry.bb_tree(msh, tdim)
    candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(msh, candidates, points_xyz)

    local_points = []
    local_cells = []
    ids = []
    values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)

    for i in range(points_xyz.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(points_xyz[i])
            local_cells.append(links[0])
            ids.append(i)

    if local_points:
        pts = np.array(local_points, dtype=np.float64)
        cells = np.array(local_cells, dtype=np.int32)
        vals = np.asarray(u_func.eval(pts, cells)).reshape(-1)
        values[np.array(ids, dtype=np.int32)] = vals

    return values


def _sample_on_output_grid(u_func: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.c_[xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    vals = _probe_function(u_func, pts).reshape(ny, nx)
    if np.isnan(vals).any():
        raise RuntimeError("Point sampling produced NaN values")
    return vals


def _solve_once(case_spec: dict, mesh_resolution: int, degree: int, dt_in: float,
                ksp_type: str, pc_type: str, rtol: float) -> dict:
    comm = MPI.COMM_WORLD
    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    n_steps = max(1, int(round((t_end - t0) / dt_in)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    kappa_spec = case_spec["pde"]["coefficients"]["kappa"]
    kappa_expr = _kappa_from_spec(kappa_spec["expr"], x)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_u_exact_expr(x, t0), V.element.interpolation_points))
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]

    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    g = fem.Function(V)
    g.interpolate(fem.Expression(_u_exact_expr(x, t0 + dt), V.element.interpolation_points))
    bc = fem.dirichletbc(g, boundary_dofs)

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(_f_exact_expr(x, t0 + dt, kappa_expr), V.element.interpolation_points))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fp.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = fp.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    start_wall = time.perf_counter()
    for step in range(n_steps):
        t = t0 + (step + 1) * dt

        g.interpolate(fem.Expression(_u_exact_expr(x, t), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(_f_exact_expr(x, t, kappa_expr), V.element.interpolation_points))

        with b.localForm() as b_local:
            b_local.set(0.0)
        fp.assemble_vector(b, L_form)
        fp.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        nit = solver.getIterationNumber()
        total_iterations += max(0, int(nit))
        u_n.x.array[:] = uh.x.array[:]

    wall = time.perf_counter() - start_wall

    u_ex_T = fem.Function(V)
    u_ex_T.interpolate(fem.Expression(_u_exact_expr(x, t_end), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array[:] - u_ex_T.x.array[:]

    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx)), op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_ex_T, u_ex_T) * ufl.dx)), op=MPI.SUM))
    rel_l2_error = l2_error / max(l2_ref, 1e-16)

    return {
        "u_h": uh,
        "u_initial": u_initial,
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
        "rel_l2_error": float(rel_l2_error),
        "wall_time": float(wall),
    }


def solve(case_spec: dict) -> dict:
    budget = 7.126
    target_error = 3.39e-3
    start = time.perf_counter()

    candidates = [
        (40, 1, 0.01),
        (56, 1, 0.0075),
        (72, 1, 0.005),
        (88, 1, 0.004),
        (104, 1, 0.003),
    ]

    best = None

    for mesh_resolution, degree, dt in candidates:
        elapsed_before = time.perf_counter() - start
        if best is not None and elapsed_before > 0.90 * budget:
            break

        try:
            current = _solve_once(case_spec, mesh_resolution, degree, dt, "cg", "hypre", 1e-10)
        except Exception:
            current = _solve_once(case_spec, mesh_resolution, degree, dt, "preonly", "lu", 1e-12)

        if best is None or current["l2_error"] < best["l2_error"]:
            best = current

        elapsed_total = time.perf_counter() - start

        if elapsed_total > 0.92 * budget:
            break
        if current["l2_error"] <= 0.15 * target_error and elapsed_total > 1.0:
            break
        if current["wall_time"] > 0.45 * budget:
            break

    if best is None:
        raise RuntimeError("Solver failed to produce a solution")

    u_grid = _sample_on_output_grid(best["u_h"], case_spec["output"]["grid"])
    u0_grid = _sample_on_output_grid(best["u_initial"], case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": str(best["time_scheme"]),
            "l2_error": float(best["l2_error"]),
            "rel_l2_error": float(best["rel_l2_error"]),
        },
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01},
            "coefficients": {
                "kappa": {
                    "type": "expr",
                    "expr": "0.2 + exp(-120*((x-0.55)**2 + (y-0.45)**2))",
                }
            },
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape, out["solver_info"])
