import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_case_time(case_spec: dict) -> Tuple[float, float, float, str]:
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", {}) if isinstance(pde.get("time", {}), dict) else {}
    t0 = float(tinfo.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(tinfo.get("t_end", case_spec.get("t_end", 0.3)))
    dt = float(tinfo.get("dt", case_spec.get("dt", 0.005)))
    scheme = str(tinfo.get("scheme", case_spec.get("scheme", "backward_euler")))
    return t0, t_end, dt, scheme


def _choose_resolution(time_budget: float = 37.853) -> Tuple[int, int, float]:
    # Use a fairly accurate default that remains safe in time.
    # Manufactured solution has strong x-growth, so P2 helps a lot.
    if time_budget > 25:
        return 104, 2, 0.0025
    return 80, 2, 0.005


def _eps_value(case_spec: dict) -> float:
    for key in ("epsilon", "eps", "diffusion", "nu"):
        if key in case_spec:
            return float(case_spec[key])
        if "pde" in case_spec and isinstance(case_spec["pde"], dict) and key in case_spec["pde"]:
            return float(case_spec["pde"][key])
    # Choose a small diffusion to align with "boundary_layer" while keeping exact source consistent.
    return 0.02


def _exact_ufl(x, t):
    return ufl.exp(-t) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.exp(4.0 * x) * np.sin(np.pi * y)


def _manufactured_source_ufl(msh, eps, t):
    x = ufl.SpatialCoordinate(msh)
    uex = _exact_ufl(x, t)
    u_t = -uex
    # Linear reaction R(u)=u
    return u_t - eps * ufl.div(ufl.grad(uex)) + uex


def _interpolate_exact(func: fem.Function, t: float):
    def exact_fun(X):
        return np.exp(-t) * np.exp(4.0 * X[0]) * np.sin(np.pi * X[1])
    func.interpolate(exact_fun)


def _sample_on_grid(domain, u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    mapping = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            mapping.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        values[np.array(mapping, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        recv = np.empty_like(values)
        comm.Allreduce(values, recv, op=MPI.MAX)
        values = recv

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    t0, t_end, dt_in, scheme = _get_case_time(case_spec)
    eps = _eps_value(case_spec)
    mesh_resolution, degree, dt_default = _choose_resolution()
    dt = min(dt_in, dt_default) if dt_in > 0 else dt_default
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"

    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(eps))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    _interpolate_exact(u_n, t0)
    u_bc = fem.Function(V)
    _interpolate_exact(u_bc, t0)

    f_ufl = _manufactured_source_ufl(domain, eps, t_const)
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_fun = fem.Function(V)
    f_fun.interpolate(f_expr)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = (u * v + dt_const * eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_const * u * v) * ufl.dx
    L = (u_n * v + dt_const * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)

    iter_total = 0
    times = np.linspace(t0, t_end, n_steps + 1)
    t_start_wall = time.perf_counter()

    initial_grid = None
    if "output" in case_spec and "grid" in case_spec["output"]:
        initial_grid = _sample_on_grid(domain, u_n, case_spec["output"]["grid"])

    for step in range(1, n_steps + 1):
        t_now = float(times[step])
        t_const.value = ScalarType(t_now)
        _interpolate_exact(u_bc, t_now)
        f_fun.interpolate(f_expr)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        iter_total += max(its, 0)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - t_start_wall

    # Accuracy verification against exact solution at t_end
    u_exact = fem.Function(V)
    _interpolate_exact(u_exact, t_end)

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    exact_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    exact_norm = math.sqrt(comm.allreduce(exact_local, op=MPI.SUM))
    rel_l2_err = l2_err / exact_norm if exact_norm > 0 else l2_err

    # Also compute sample-grid max error if output grid provided
    out_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]})
    u_grid = _sample_on_grid(domain, uh, out_grid)
    xs = np.linspace(out_grid["bbox"][0], out_grid["bbox"][1], int(out_grid["nx"]))
    ys = np.linspace(out_grid["bbox"][2], out_grid["bbox"][3], int(out_grid["ny"]))
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = _exact_numpy(XX, YY, t_end)
    grid_max_err = float(np.max(np.abs(u_grid - u_exact_grid)))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1.0e-10,
        "iterations": int(iter_total),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification_l2_error": float(l2_err),
        "verification_rel_l2_error": float(rel_l2_err),
        "verification_grid_max_error": float(grid_max_err),
        "wall_time_observed": float(wall),
        "epsilon": float(eps),
    }

    result = {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": initial_grid if initial_grid is not None else _sample_on_grid(domain, u_n, out_grid),
    }

    if rank == 0:
        return result
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.3,
                "dt": 0.005,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "epsilon": 0.02,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
