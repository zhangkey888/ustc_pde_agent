from __future__ import annotations

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _build_mixed_space(msh, degree_u: int = 2, degree_p: int = 1):
    cell_name = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _exact_velocity_expr(x):
    t = ufl.tanh(6 * (x[0] - 0.5))
    return ufl.as_vector(
        [
            ufl.pi * t * ufl.cos(ufl.pi * x[1]),
            -6 * (1 - t**2) * ufl.sin(ufl.pi * x[1]),
        ]
    )


def _exact_pressure_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])




def _all_boundary_facets(msh):
    fdim = msh.topology.dim - 1
    return mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))


def _make_velocity_bc(W, V, msh):
    x = ufl.SpatialCoordinate(msh)
    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(_exact_velocity_expr(x), V.element.interpolation_points))
    facets = _all_boundary_facets(msh)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim - 1, facets)
    return fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))


def _make_pressure_pin_bc(W, Q):
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    return fem.dirichletbc(p0, p_dofs, W.sub(1))






def _solve_ns(
    n: int,
    degree_u: int = 2,
    degree_p: int = 1,
    nu_value: float = 0.16,
    newton_rtol: float = 1e-10,
    newton_atol: float = 1e-12,
    newton_max_it: int = 20,
):
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_mixed_space(msh, degree_u, degree_p)
    x = ufl.SpatialCoordinate(msh)
    zero_vec = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=np.float64))

    bc_u = _make_velocity_bc(W, V, msh)
    bc_p = _make_pressure_pin_bc(W, Q)
    bcs = [bc_u, bc_p]

    w = fem.Function(W)
    w.x.array[:] = 0.0
    w.x.scatter_forward()
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    nu = ScalarType(nu_value)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(zero_vec, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": newton_atol,
            "snes_max_it": newton_max_it,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-10,
            "pc_type": "lu",
        },
    )

    t0 = time.perf_counter()
    problem.solve()
    solve_time = time.perf_counter() - t0
    w.x.scatter_forward()

    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(_exact_velocity_expr(x), V.element.interpolation_points))
    p_exact_fun = fem.Function(Q)
    p_exact_fun.interpolate(fem.Expression(_exact_pressure_expr(x), Q.element.interpolation_points))

    err_u_L2 = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(uh - u_exact_fun, uh - u_exact_fun) * ufl.dx)),
        op=MPI.SUM,
    )
    err_p_L2 = msh.comm.allreduce(
        fem.assemble_scalar(fem.form((ph - p_exact_fun) * (ph - p_exact_fun) * ufl.dx)),
        op=MPI.SUM,
    )
    err_u_H1 = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh - u_exact_fun), ufl.grad(uh - u_exact_fun)) * ufl.dx)),
        op=MPI.SUM,
    )

    snes = problem.solver
    ksp = snes.getKSP()
    rtol = ksp.getTolerances()[0]
    if rtol is None:
        rtol = 1e-10

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree_u),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "nonlinear_iterations": [int(snes.getIterationNumber())],
        "accuracy_verification": {
            "velocity_L2_error": float(math.sqrt(max(err_u_L2, 0.0))),
            "velocity_H1_semi_error": float(math.sqrt(max(err_u_H1, 0.0))),
            "pressure_L2_error": float(math.sqrt(max(err_p_L2, 0.0))),
            "wall_time_sec": float(solve_time),
        },
    }
    return uh, info


def _sample_velocity_magnitude(u_fun: fem.Function, grid: dict) -> np.ndarray:
    msh = u_fun.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idxs, dtype=np.int32), :] = np.real(vals)

    gathered = msh.comm.allgather(local_vals)
    vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(vals[:, 0]) & ~np.isnan(arr[:, 0])
        vals[mask, :] = arr[mask, :]

    mag = np.linalg.norm(vals, axis=1)
    mag = np.nan_to_num(mag, nan=0.0)
    return mag.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]

    budget = None
    if "time_limit_sec" in case_spec:
        budget = float(case_spec["time_limit_sec"])
    elif "wall_time_sec" in case_spec:
        budget = float(case_spec["wall_time_sec"])
    elif isinstance(case_spec.get("pde", {}).get("time"), dict):
        tl = case_spec["pde"]["time"].get("time_limit_sec", None)
        if tl is not None:
            budget = float(tl)

    n = 80
    if budget is not None:
        if budget > 200:
            n = 112
        elif budget > 80:
            n = 96

    uh, info = _solve_ns(n=n, degree_u=2, degree_p=1, nu_value=0.16)
    u_grid = _sample_velocity_magnitude(uh, output_grid)
    return {"u": u_grid, "solver_info": info}


if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 16, "ny": 12, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
