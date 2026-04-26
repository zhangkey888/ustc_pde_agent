import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_function(u_func: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points_xyz)
    values = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)

    pts_local = []
    cells_local = []
    idx_local = []
    for i in range(points_xyz.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points_xyz[i])
            cells_local.append(links[0])
            idx_local.append(i)

    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        values[np.array(idx_local, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = msh.comm.allgather(values)
    out = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    return out


def _sample_grid(u_func: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, points)
    return vals.reshape(ny, nx)


def _solve_level(case_spec: dict, mesh_resolution: int, degree: int, dt: float,
                 eps_value: float = 0.1, ksp_type: str = "cg", pc_type: str = "hypre",
                 rtol: float = 1.0e-10) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.4))
    scheme = str(pde_time.get("scheme", "backward_euler"))
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(eps_value))

    u_exact = ufl.exp(-t_c) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_exact = ufl.exp(-t_c) * ufl.exp(x[0]) * (1.0 - ufl.pi**2) * ufl.sin(ufl.pi * x[1])
    u_t_exact = -ufl.exp(-t_c) * ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    reaction_exact = u_exact
    f_expr = u_t_exact - eps_c * lap_u_exact + reaction_exact

    u_bc = fem.Function(V)
    f_fun = fem.Function(V)
    u_n = fem.Function(V)
    uh = fem.Function(V)
    u_ex_fun = fem.Function(V)

    def update_time_dependent_fields(t: float):
        t_c.value = ScalarType(t)
        expr_u = fem.Expression(u_exact, V.element.interpolation_points)
        expr_f = fem.Expression(f_expr, V.element.interpolation_points)
        u_bc.interpolate(expr_u)
        f_fun.interpolate(expr_f)
        u_ex_fun.interpolate(expr_u)

    update_time_dependent_fields(t0)
    u_n.x.array[:] = u_ex_fun.x.array
    u_n.x.scatter_forward()
    uh.x.array[:] = u_n.x.array
    uh.x.scatter_forward()

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v / dt_c + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx
    L = (u_n * v / dt_c + f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
    except Exception:
        pass

    initial_grid = _sample_grid(u_n, case_spec["output"]["grid"])

    total_iterations = 0
    nonlinear_iterations = []
    t = t0

    for _ in range(n_steps):
        t += dt
        update_time_dependent_fields(t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            converged = solver.getConvergedReason() > 0
            if not converged:
                raise RuntimeError("Iterative solver did not converge")
            its = int(solver.getIterationNumber())
        except Exception:
            fallback = PETSc.KSP().create(comm)
            fallback.setOperators(A)
            fallback.setType("preonly")
            fallback.getPC().setType("lu")
            fallback.setTolerances(rtol=rtol)
            fallback.solve(b, uh.x.petsc_vec)
            solver = fallback
            its = 1

        total_iterations += its
        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
        nonlinear_iterations.append(1)

    update_time_dependent_fields(t_end)
    err_sq = fem.assemble_scalar(fem.form((uh - u_ex_fun) * (uh - u_ex_fun) * ufl.dx))
    ex_sq = fem.assemble_scalar(fem.form(u_ex_fun * u_ex_fun * ufl.dx))
    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    ex_sq = comm.allreduce(ex_sq, op=MPI.SUM)
    l2_error = math.sqrt(max(err_sq, 0.0))
    rel_l2_error = l2_error / math.sqrt(max(ex_sq, 1.0e-30))

    result = {
        "u": _sample_grid(uh, case_spec["output"]["grid"]),
        "u_initial": initial_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(solver.getType()),
            "pc_type": str(solver.getPC().getType()),
            "rtol": float(rtol),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": str(scheme),
            "nonlinear_iterations": nonlinear_iterations,
            "l2_error": float(l2_error),
            "relative_l2_error": float(rel_l2_error),
        },
    }

    if rank == 0:
        result["solver_info"]["verification"] = {
            "manufactured_solution": "exp(-t)*exp(x)*sin(pi*y)",
            "reaction_term": "R(u)=u",
            "passed_target_8e-3": bool(rel_l2_error <= 8.0e-3),
        }
    else:
        result["solver_info"]["verification"] = {
            "manufactured_solution": "exp(-t)*exp(x)*sin(pi*y)",
            "reaction_term": "R(u)=u",
            "passed_target_8e-3": bool(rel_l2_error <= 8.0e-3),
        }
    return result


def solve(case_spec: dict) -> dict:
    pde_time = case_spec.get("pde", {}).get("time", {})
    suggested_dt = float(pde_time.get("dt", 0.01))
    dt = min(suggested_dt, 0.01)
    return _solve_level(case_spec, mesh_resolution=20, degree=1, dt=dt)


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.4, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 25, "ny": 21, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print("u shape:", out["u"].shape)
        print("l2 error:", out["solver_info"]["l2_error"])
        print("relative l2 error:", out["solver_info"]["relative_l2_error"])
        print("wall:", wall)
