import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
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
# pde_skill: reaction_diffusion
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _parse_case(case_spec):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) if isinstance(pde.get("time", {}), dict) else {}
    t0 = float(time_spec.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.4)))
    dt = float(time_spec.get("dt", case_spec.get("dt", 0.02)))
    scheme = time_spec.get("scheme", case_spec.get("scheme", "backward_euler"))
    eps = float(pde.get("epsilon", case_spec.get("epsilon", 0.1)))
    reaction_coeff = float(pde.get("reaction_coefficient", case_spec.get("reaction_coefficient", 1.0)))
    return t0, t_end, dt, scheme, eps, reaction_coeff


def _make_exact_and_forcing(msh, eps, reaction_coeff):
    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, ScalarType(0.0))
    u_exact = ufl.exp(-t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # u_t - eps Δu + r u = f, with Δsin(pi x)sin(pi y) = -2 pi^2 sin sin
    f_expr = (-1.0 + 2.0 * eps * ufl.pi**2 + reaction_coeff) * u_exact
    return t_c, u_exact, f_expr


def _boundary_all(x):
    return np.logical_or.reduce(
        (
            np.isclose(x[0], 0.0),
            np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], 1.0),
        )
    )


def _probe_function(u_func, pts3):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3.T)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3.T)

    values = np.full((pts3.shape[1],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts3.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3.T[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if msh.comm.size > 1:
        recv = np.empty_like(values)
        msh.comm.Allreduce(values, recv, op=MPI.SUM)
        mask_local = np.isfinite(values).astype(np.int32)
        mask_global = np.empty_like(mask_local)
        msh.comm.Allreduce(mask_local, mask_global, op=MPI.SUM)
        good = mask_global > 0
        recv[~good] = np.nan
        return recv
    return values


def _sample_to_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    vals = _probe_function(u_func, pts3)
    return vals.reshape((ny, nx))


def _l2_error(u_h, u_exact_expr):
    msh = u_h.function_space.mesh
    diff = u_h - u_exact_expr
    err_form = fem.form(ufl.inner(diff, diff) * ufl.dx)
    val = fem.assemble_scalar(err_form)
    val = msh.comm.allreduce(val, op=MPI.SUM)
    return math.sqrt(max(val, 0.0))


def _solve_candidate(case_spec, nx, degree, dt, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    t0, t_end, _, scheme, eps, reaction_coeff = _parse_case(case_spec)
    assert scheme == "backward_euler"

    msh = mesh.create_unit_square(COMM, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    t_c, u_exact_expr, f_expr = _make_exact_and_forcing(msh, eps, reaction_coeff)

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_sol = fem.Function(V)
    u_sol.name = "u"

    t_c.value = ScalarType(t0)
    u_n.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    u0_grid = None

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, _boundary_all)
    bc_fun = fem.Function(V)
    bc_fun.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(bc_fun, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps))
    react_c = fem.Constant(msh, ScalarType(reaction_coeff))

    a = ((1.0 / dt_c) * u * v + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) + react_c * u * v) * ufl.dx
    L = ((1.0 / dt_c) * u_n * v + f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    t = t0
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    dt_c.value = ScalarType(dt)

    nonlinear_iterations = []
    total_iterations = 0

    for step in range(1, n_steps + 1):
        t = t0 + step * dt
        t_c.value = ScalarType(t)
        bc_fun.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_sol.x.petsc_vec)
            reason = solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"KSP failed with reason {reason}")
        except Exception:
            solver = PETSc.KSP().create(msh.comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, u_sol.x.petsc_vec)

        u_sol.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its)
        nonlinear_iterations.append(1)
        u_n.x.array[:] = u_sol.x.array

    err = _l2_error(u_sol, u_exact_expr)
    return {
        "mesh": msh,
        "V": V,
        "u": u_sol,
        "u_exact_expr": u_exact_expr,
        "error_l2": err,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": scheme,
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "nonlinear_iterations": nonlinear_iterations,
        "t_final": t,
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    t0, t_end, dt_user, scheme, eps, reaction_coeff = _parse_case(case_spec)

    grid = case_spec["output"]["grid"]
    budget = 43.015
    target_soft = min(0.92 * budget, 38.0)

    candidates = [
        (64, 2, min(dt_user, 0.01)),
    ]

    best = None
    for nx, degree, dt in candidates:
        elapsed = time.perf_counter() - start
        if elapsed > target_soft and best is not None:
            break
        result = _solve_candidate(case_spec, nx=nx, degree=degree, dt=dt)
        if best is None or result["error_l2"] < best["error_l2"]:
            best = result
        if result["error_l2"] <= 2.78e-2 and elapsed > 0.35 * budget:
            break

    if best is None:
        best = _solve_candidate(case_spec, nx=40, degree=1, dt=dt_user)

    msh = best["mesh"]
    V = best["V"]
    u_sol = best["u"]

    # Build initial condition again for requested output
    t_c, u_exact_expr, _ = _make_exact_and_forcing(msh, eps, reaction_coeff)
    t_c.value = ScalarType(t0)
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    u_grid = _sample_to_grid(u_sol, grid)
    u_init_grid = _sample_to_grid(u_init, grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "dt": float(best["dt"]),
        "n_steps": int(best["n_steps"]),
        "time_scheme": str(best["time_scheme"]),
        "nonlinear_iterations": [int(k) for k in best["nonlinear_iterations"]],
        "l2_error_vs_manufactured": float(best["error_l2"]),
        "epsilon": float(eps),
        "reaction_coefficient": float(reaction_coeff),
        "wall_time_sec": float(time.perf_counter() - start),
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.4, "dt": 0.02, "scheme": "backward_euler"},
            "epsilon": 0.1,
            "reaction_coefficient": 1.0,
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if COMM.rank == 0:
        print("L2_ERROR:", out["solver_info"]["l2_error_vs_manufactured"])
        print("WALL_TIME:", out["solver_info"]["wall_time_sec"])
        print(out["u"].shape)
        print(out["solver_info"])
