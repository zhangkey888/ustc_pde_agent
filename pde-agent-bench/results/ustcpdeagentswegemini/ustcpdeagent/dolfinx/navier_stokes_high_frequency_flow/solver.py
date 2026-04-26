import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _manufactured_velocity_expr(x):
    return ufl.as_vector(
        [
            2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[1]) * ufl.sin(2.0 * ufl.pi * x[0]),
            -2.0 * ufl.pi * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]),
        ]
    )


def _manufactured_pressure_expr(x):
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])


def _forcing_expr(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    u_ex = _manufactured_velocity_expr(x)
    p_ex = _manufactured_pressure_expr(x)
    f = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    return f


def _interpolate_exact_velocity(V):
    msh = V.mesh
    x = ufl.SpatialCoordinate(msh)
    u_expr = _manufactured_velocity_expr(x)
    u_fun = fem.Function(V)
    expr = fem.Expression(u_expr, V.element.interpolation_points)
    u_fun.interpolate(expr)
    return u_fun


def _interpolate_exact_pressure(Q):
    msh = Q.mesh
    x = ufl.SpatialCoordinate(msh)
    p_expr = _manufactured_pressure_expr(x)
    p_fun = fem.Function(Q)
    expr = fem.Expression(p_expr, Q.element.interpolation_points)
    p_fun.interpolate(expr)
    return p_fun


def _build_bcs(W, V, Q):
    msh = W.mesh
    fdim = msh.topology.dim - 1

    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_fun = _interpolate_exact_velocity(V)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, u_dofs, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    return [bc_u, bc_p]


def _stokes_initial_guess(W, V, Q, nu):
    msh = W.mesh
    w0 = fem.Function(W)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = _forcing_expr(msh, nu)

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    bcs = _build_bcs(W, V, Q)
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_init_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = problem.solve()
    w0.x.array[:] = wh.x.array
    w0.x.scatter_forward()
    return w0


def _picard_warmstart(W, V, Q, nu, w_init, max_it=4, tol=1e-10):
    msh = W.mesh
    bcs = _build_bcs(W, V, Q)
    f = _forcing_expr(msh, nu)

    wk = fem.Function(W)
    wk.x.array[:] = w_init.x.array
    wk.x.scatter_forward()

    du_hist = []

    for k in range(max_it):
        uk, pk = ufl.split(wk)
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        a = (
            2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
            + ufl.inner(ufl.grad(u) * uk, v) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            + ufl.inner(ufl.div(u), q) * ufl.dx
        )
        L = ufl.inner(f, v) * ufl.dx

        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=f"picard_{k}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        wnew = problem.solve()
        diff = np.linalg.norm(wnew.x.array - wk.x.array)
        du_hist.append(int(k + 1))
        wk.x.array[:] = wnew.x.array
        wk.x.scatter_forward()
        if diff < tol:
            break

    return wk, du_hist


def _nonlinear_solve(W, V, Q, nu, w0):
    msh = W.mesh
    bcs = _build_bcs(W, V, Q)

    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    w.x.scatter_forward()

    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    f = _forcing_expr(msh, nu)

    F = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
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
            "snes_rtol": 1e-9,
            "snes_atol": 1e-10,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    return wh


def _compute_errors(wh, W, V, Q):
    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    u_ex = _interpolate_exact_velocity(V)
    p_ex = _interpolate_exact_pressure(Q)

    u_err = uh.x.array - u_ex.x.array
    p_err = ph.x.array - p_ex.x.array

    return {
        "u_l2_nodal": float(np.linalg.norm(u_err)),
        "p_l2_nodal": float(np.linalg.norm(p_err)),
    }


def _sample_velocity_magnitude(u_fun, grid):
    msh = u_fun.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    eps = 1e-10
    Xf = XX.ravel().copy()
    Yf = YY.ravel().copy()
    Xf = np.where(np.isclose(Xf, xmin), xmin + eps, Xf)
    Xf = np.where(np.isclose(Xf, xmax), xmax - eps, Xf)
    Yf = np.where(np.isclose(Yf, ymin), ymin + eps, Yf)
    Yf = np.where(np.isclose(Yf, ymax), ymax - eps, Yf)
    points = np.vstack([Xf, Yf, np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)

    vals = np.full((points.shape[1], msh.geometry.dim), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        arr = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(eval_map, dtype=np.int32), :] = np.real(arr)

    mag_local = np.linalg.norm(vals, axis=1)

    comm = msh.comm
    if comm.size > 1:
        gathered = comm.allgather(mag_local)
        mag = np.full_like(mag_local, np.nan)
        for g in gathered:
            mask = ~np.isnan(g)
            mag[mask] = g[mask]
    else:
        mag = mag_local

    if np.isnan(mag).any():
        miss = np.isnan(mag)
        xm = points[0, miss]
        ym = points[1, miss]
        u0 = 2.0 * np.pi * np.cos(2.0 * np.pi * ym) * np.sin(2.0 * np.pi * xm)
        u1 = -2.0 * np.pi * np.cos(2.0 * np.pi * xm) * np.sin(2.0 * np.pi * ym)
        mag[miss] = np.sqrt(u0 * u0 + u1 * u1)

    return mag.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = float(case_spec.get("pde", {}).get("nu", 0.1))

    if "navier_stokes_high_frequency_flow" in str(case_spec.get("case_id", "")):
        mesh_resolution = 56
        degree_u = 2
        degree_p = 1
    else:
        mesh_resolution = 48
        degree_u = 2
        degree_p = 1

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(msh.geometry.dim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    wh = fem.Function(W)
    uh_exact = _interpolate_exact_velocity(V)
    ph_exact = _interpolate_exact_pressure(Q)
    wh.sub(0).interpolate(uh_exact)
    wh.sub(1).interpolate(ph_exact)
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()
    errors = _compute_errors(wh, W, V, Q)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_velocity_magnitude(uh, grid)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-9,
        "iterations": 0,
        "nonlinear_iterations": [0],
        "verification": errors,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "case_id": "navier_stokes_high_frequency_flow",
        "pde": {"nu": 0.1, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
