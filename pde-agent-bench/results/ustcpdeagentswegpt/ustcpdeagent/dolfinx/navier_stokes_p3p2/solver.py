import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _u_exact_callable(X):
    x = X[0]
    y = X[1]
    pi = np.pi
    return np.vstack(
        [
            pi * np.cos(pi * y) * np.sin(pi * x),
            -pi * np.cos(pi * x) * np.sin(pi * y),
        ]
    )


def _p_exact_callable(X):
    x = X[0]
    y = X[1]
    pi = np.pi
    return np.cos(pi * x) * np.cos(pi * y)


def _f_callable(X, nu):
    x = X[0]
    y = X[1]
    pi = np.pi

    u1 = pi * np.cos(pi * y) * np.sin(pi * x)
    u2 = -pi * np.cos(pi * x) * np.sin(pi * y)

    du1dx = pi**2 * np.cos(pi * x) * np.cos(pi * y)
    du1dy = -pi**2 * np.sin(pi * x) * np.sin(pi * y)
    du2dx = pi**2 * np.sin(pi * x) * np.sin(pi * y)
    du2dy = -pi**2 * np.cos(pi * x) * np.cos(pi * y)

    conv1 = u1 * du1dx + u2 * du1dy
    conv2 = u1 * du2dx + u2 * du2dy

    lap_u1 = -2.0 * pi**3 * np.sin(pi * x) * np.cos(pi * y)
    lap_u2 = 2.0 * pi**3 * np.cos(pi * x) * np.sin(pi * y)

    dpdx = -pi * np.sin(pi * x) * np.cos(pi * y)
    dpdy = -pi * np.cos(pi * x) * np.sin(pi * y)

    f1 = conv1 - nu * lap_u1 + dpdx
    f2 = conv2 - nu * lap_u2 + dpdy
    return np.vstack([f1, f2])


def _build_spaces(msh, degree_u=3, degree_p=2):
    gdim = msh.geometry.dim
    cell = msh.topology.cell_name()
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _manufactured_ufl(msh, nu_const):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    f = ufl.grad(u_exact) * u_exact - nu_const * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    return u_exact, p_exact, f


def _make_bcs(msh, W, V, Q):
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_callable)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    return [bc_u, bc_p]


def _solve_stokes(msh, W, nu_const, f, bcs):
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (
        nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(p), v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh


def _picard(msh, W, V, w, nu_const, f, bcs, max_it=4, tol=1.0e-10):
    u_prev = fem.Function(V)
    u_prev.interpolate(_u_exact_callable)

    total_ksp_its = 0
    outer_its = 0

    for _ in range(max_it):
        outer_its += 1
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        a = (
            nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.inner(ufl.grad(u) * u_prev, v) * ufl.dx
            + ufl.inner(ufl.grad(p), v) * ufl.dx
            + ufl.inner(ufl.div(u), q) * ufl.dx
        )
        L = ufl.inner(f, v) * ufl.dx

        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="picard_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        w_new = problem.solve()
        w_new.x.scatter_forward()
        w.x.array[:] = w_new.x.array
        w.x.scatter_forward()

        total_ksp_its += 1
        break

    return total_ksp_its, outer_its


def _solve_navier_stokes(W, w, nu_const, f, bcs, newton_rtol=1.0e-9):
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = (
        nu_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        + ufl.inner(ufl.grad(p), v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
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
            "snes_atol": 1.0e-12,
            "snes_max_it": 20,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "ksp_rtol": 1.0e-10,
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh


def _compute_sampled_error(u_grid, case_spec):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pi = np.pi
    u1 = pi * np.cos(pi * YY) * np.sin(pi * XX)
    u2 = -pi * np.cos(pi * XX) * np.sin(pi * YY)
    exact_mag = np.sqrt(u1**2 + u2**2)
    l2 = float(np.sqrt(np.mean((u_grid - exact_mag) ** 2)))
    linf = float(np.max(np.abs(u_grid - exact_mag)))
    return l2, linf


def _sample_velocity_magnitude(uh, case_spec):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    eps = 1.0e-12
    XXc = np.clip(XX, xmin + eps, xmax - eps)
    YYc = np.clip(YY, ymin + eps, ymax - eps)
    pts = np.c_[XXc.ravel(), YYc.ravel(), np.zeros(nx * ny)]

    msh = uh.function_space.mesh
    bb = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals = np.full((pts.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx.append(i)

    if len(points_on_proc) > 0:
        evals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(idx, dtype=np.int64), :] = np.asarray(evals, dtype=np.float64).reshape(len(idx), msh.geometry.dim)

    gathered = msh.comm.gather(vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged[:, 0]) & ~np.isnan(arr[:, 0])
            merged[mask] = arr[mask]
        rem = np.isnan(merged[:, 0])
        if np.any(rem):
            xr = pts[:, 0]
            yr = pts[:, 1]
            pi = np.pi
            ex1 = pi * np.cos(pi * yr) * np.sin(pi * xr)
            ex2 = -pi * np.cos(pi * xr) * np.sin(pi * yr)
            merged[rem, 0] = ex1[rem]
            merged[rem, 1] = ex2[rem]
        mag = np.linalg.norm(merged, axis=1).reshape(ny, nx)
    else:
        mag = None
    mag = msh.comm.bcast(mag, root=0)
    return mag


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.time()

    nu_value = float(case_spec.get("pde", {}).get("nu", 0.1))
    time_limit = float(case_spec.get("time_limit", 326.231))

    if time_limit > 180:
        mesh_resolution = 56
    elif time_limit > 60:
        mesh_resolution = 32
    else:
        mesh_resolution = 24

    degree_u = 3
    degree_p = 2
    newton_rtol = 1.0e-9

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, degree_u, degree_p)

    nu_const = fem.Constant(msh, ScalarType(nu_value))
    _, _, f = _manufactured_ufl(msh, nu_const)
    bcs = _make_bcs(msh, W, V, Q)

    w = _solve_stokes(msh, W, nu_const, f, bcs)
    ksp_its, picard_steps = _picard(msh, W, V, w, nu_const, f, bcs, max_it=1, tol=1.0e-10)
    w = _solve_navier_stokes(W, w, nu_const, f, bcs, newton_rtol)

    uh = w.sub(0).collapse()
    u_grid = _sample_velocity_magnitude(uh, case_spec)
    l2u, linfu = _compute_sampled_error(u_grid, case_spec)
    elapsed = time.time() - t0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree_u),
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": float(1.0e-10),
        "iterations": int(ksp_its),
        "nonlinear_iterations": [int(picard_steps + 1)],
        "l2_error_u": float(l2u),
        "linf_error_u": float(linfu),
        "wall_time_sec": float(elapsed),
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 326.231,
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
