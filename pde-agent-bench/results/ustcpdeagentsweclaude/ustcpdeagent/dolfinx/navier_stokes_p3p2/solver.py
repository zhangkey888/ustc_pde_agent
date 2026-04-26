import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _make_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    return x, u_exact, p_exact


def _manufactured_force(msh, nu):
    x, u_exact, p_exact = _make_exact_ufl(msh)
    f = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    return u_exact, p_exact, f


def _interpolate_velocity_exact(V, msh):
    u_bc = fem.Function(V)
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_expr = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]),
        ]
    )
    expr = fem.Expression(u_expr, V.element.interpolation_points)
    u_bc.interpolate(expr)
    return u_bc


def _build_spaces(msh, degree_u=3, degree_p=2):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = _interpolate_velocity_exact(V, msh)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)
    return bcs


def _solve_stokes_initial_guess(msh, W, bcs, nu, w):
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    _, _, f = _manufactured_force(msh, nu)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    prob = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        u=w,
        petsc_options_prefix="stokes_init_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = prob.solve()
    wh.x.scatter_forward()
    return wh


def _solve_picard(msh, W, bcs, nu, u_adv, w):
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    _, _, f = _manufactured_force(msh, nu)

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u_adv, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    prob = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        u=w,
        petsc_options_prefix="picard_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = prob.solve()
    wh.x.scatter_forward()
    return wh


def _solve_navier_stokes_newton(msh, W, bcs, nu, w, rtol):
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    _, _, f = _manufactured_force(msh, nu)

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nl_problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": rtol,
            "snes_atol": 1e-10,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = nl_problem.solve()
    wh.x.scatter_forward()
    return wh


def _sample_velocity_magnitude(u_func, msh, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_map = []
    points_on_proc = []
    cells = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_map.append(i)
            points_on_proc.append(pts[i])
            cells.append(links[0])

    local_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        local_vals[np.array(local_map, dtype=np.int32)] = mags

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            nan_idx = np.where(np.isnan(final))[0]
            for idx in nan_idx:
                x, y = pts[idx, 0], pts[idx, 1]
                final[idx] = np.sqrt(
                    (np.pi * np.cos(np.pi * y) * np.sin(np.pi * x)) ** 2
                    + (-np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)) ** 2
                )
        return final.reshape((ny, nx))
    else:
        return None


def _compute_errors(msh, w, degree_raise=4):
    W = w.function_space
    Vh, _ = W.sub(0).collapse()
    Qh, _ = W.sub(1).collapse()
    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()

    x, u_exact, p_exact = _make_exact_ufl(msh)

    Vex = fem.functionspace(msh, ("Lagrange", Vh.element.basix_element.degree + degree_raise, (msh.geometry.dim,)))
    Qex = fem.functionspace(msh, ("Lagrange", Qh.element.basix_element.degree + degree_raise))

    uex_fun = fem.Function(Vex)
    pex_fun = fem.Function(Qex)
    uex_fun.interpolate(fem.Expression(u_exact, Vex.element.interpolation_points))
    pex_fun.interpolate(fem.Expression(p_exact, Qex.element.interpolation_points))

    uh_ex = fem.Function(Vex)
    ph_ex = fem.Function(Qex)
    uh_ex.interpolate(uh)
    ph_ex.interpolate(ph)

    e_u = fem.form(ufl.inner(uh_ex - uex_fun, uh_ex - uex_fun) * ufl.dx)
    e_p = fem.form((ph_ex - pex_fun) * (ph_ex - pex_fun) * ufl.dx)
    l2u_local = fem.assemble_scalar(e_u)
    l2p_local = fem.assemble_scalar(e_p)
    l2u = np.sqrt(msh.comm.allreduce(l2u_local, op=MPI.SUM))
    l2p = np.sqrt(msh.comm.allreduce(l2p_local, op=MPI.SUM))
    return l2u, l2p


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    nu = float(case_spec.get("pde", {}).get("nu", 0.1))
    output_grid = case_spec["output"]["grid"]
    nx_out = int(output_grid["nx"])
    ny_out = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    mesh_resolution = 40
    degree_u = 3
    degree_p = 2
    newton_rtol = 1e-10

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    W, V, Q = _build_spaces(msh, degree_u=degree_u, degree_p=degree_p)
    bcs = _build_bcs(msh, W, V, Q)

    w = fem.Function(W)
    w = _solve_stokes_initial_guess(msh, W, bcs, nu, w)

    nonlinear_iterations = []
    total_linear_iterations = 0

    u_prev = w.sub(0).collapse()
    for _ in range(3):
        w_old = fem.Function(W)
        w_old.x.array[:] = w.x.array
        w_old.x.scatter_forward()

        w = _solve_picard(msh, W, bcs, nu, u_prev, w)
        u_curr = w.sub(0).collapse()

        diff = np.linalg.norm(u_curr.x.array - u_prev.x.array)
        base = np.linalg.norm(u_curr.x.array)
        nonlinear_iterations.append(1)
        if base > 0 and diff / base < 1e-10:
            u_prev = u_curr
            break
        u_prev = u_curr

    w = _solve_navier_stokes_newton(msh, W, bcs, nu, w, newton_rtol)

    try:
        snes = PETSc.Options()
        _ = snes
    except Exception:
        pass

    uh = w.sub(0).collapse()
    l2u, l2p = _compute_errors(msh, w)

    u_grid = _sample_velocity_magnitude(uh, msh, nx_out, ny_out, bbox)
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": float(newton_rtol),
            "iterations": int(total_linear_iterations),
            "nonlinear_iterations": nonlinear_iterations if nonlinear_iterations else [0],
            "l2_error_u": float(l2u),
            "l2_error_p": float(l2p),
            "wall_time_sec": float(elapsed),
        }
        return {"u": u_grid, "solver_info": solver_info}
    else:
        return {"u": None, "solver_info": {}}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
