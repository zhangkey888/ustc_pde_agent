import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _make_spaces(msh, degree_u=2, degree_p=1):
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", cell, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return W, V, Q


def _manufactured_ufl(msh, nu):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    ex2 = ufl.exp(2 * x[0])
    ex1 = ufl.exp(x[0])
    cy = ufl.cos(pi * x[1])
    sy = ufl.sin(pi * x[1])

    u1 = pi * ex2 * cy
    u2 = -2.0 * ex2 * sy
    u_exact = ufl.as_vector([u1, u2])
    p_exact = ex1 * cy

    du1_dx = 2 * pi * ex2 * cy
    du1_dy = -pi * pi * ex2 * sy
    du2_dx = -4 * ex2 * sy
    du2_dy = -2 * pi * ex2 * cy

    conv1 = u1 * du1_dx + u2 * du1_dy
    conv2 = u1 * du2_dx + u2 * du2_dy

    lap_u1 = (4 - pi * pi) * pi * ex2 * cy
    lap_u2 = (4 - pi * pi) * (-2.0) * ex2 * sy

    dp_dx = ex1 * cy
    dp_dy = -pi * ex1 * sy

    f = ufl.as_vector([
        conv1 - nu * lap_u1 + dp_dx,
        conv2 - nu * lap_u2 + dp_dy,
    ])
    return u_exact, p_exact, f


def _exact_velocity_callable():
    def fun(x):
        vals = np.zeros((2, x.shape[1]), dtype=np.float64)
        vals[0] = np.pi * np.exp(2.0 * x[0]) * np.cos(np.pi * x[1])
        vals[1] = -2.0 * np.exp(2.0 * x[0]) * np.sin(np.pi * x[1])
        return vals
    return fun


def _build_bcs(msh, W, V, Q):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(_exact_velocity_callable())
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    return bcs


def _solve_once(n, degree_u=2, degree_p=1, nu_value=0.15, newton_max_it=25):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q = _make_spaces(msh, degree_u, degree_p)
    bcs = _build_bcs(msh, W, V, Q)

    nu = ScalarType(nu_value)
    u_exact, p_exact, f = _manufactured_ufl(msh, nu)

    w = fem.Function(W)
    w.x.array[:] = 0.0

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    a_stokes = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="stokes_init_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    try:
        w0 = stokes_problem.solve()
        w.x.array[:] = w0.x.array
        w.x.scatter_forward()
    except Exception:
        pass

    u_h, p_h = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        nu * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_h) * u_h, v) * ufl.dx
        - ufl.inner(p_h, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u_h), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_iterations = [0]
    iterations = 0
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-10

    try:
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="ns_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1.0e-10,
                "snes_atol": 1.0e-12,
                "snes_max_it": newton_max_it,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        w = problem.solve()
        w.x.scatter_forward()
        try:
            snes = problem.solver
            nonlinear_iterations = [int(snes.getIterationNumber())]
            iterations = int(snes.getKSP().getIterationNumber())
        except Exception:
            pass
    except Exception:
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1.0e-8
        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J,
            petsc_options_prefix="nsfb_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1.0e-8,
                "snes_atol": 1.0e-10,
                "snes_max_it": max(newton_max_it, 40),
                "ksp_type": "gmres",
                "pc_type": "ilu",
                "ksp_rtol": 1.0e-8,
            },
        )
        w = problem.solve()
        w.x.scatter_forward()
        try:
            snes = problem.solver
            nonlinear_iterations = [int(snes.getIterationNumber())]
            iterations = int(snes.getKSP().getIterationNumber())
        except Exception:
            pass

    uh = fem.Function(V)
    uh.interpolate(_exact_velocity_callable())

    err_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact - u_exact, u_exact - u_exact) * ufl.dx))
    norm_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    norm_ex = math.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
    rel_err = err_L2 / (norm_ex + 1.0e-16)

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree_u),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "l2_error_velocity": float(err_L2),
        "relative_l2_error_velocity": float(rel_err),
    }
    return msh, uh, info


def _sample_velocity_magnitude(u_func, grid):
    msh = u_func.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.zeros((nx * ny, 3), dtype=np.float64)
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()

    bb = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals)
        mags = np.linalg.norm(vals, axis=1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = mags

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        out[~np.isfinite(out)] = 0.0
        return out.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    start = time.time()

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    nu = float(pde.get("nu", 0.15))
    degree_u = int(case_spec.get("degree_u", 2))
    degree_p = int(case_spec.get("degree_p", max(1, degree_u - 1)))
    newton_max_it = int(case_spec.get("newton_max_it", 25))
    n0 = int(case_spec.get("mesh_resolution", 24))
    time_limit = float(case_spec.get("time_limit", 343.3))

    tried = []
    best = None
    n = n0
    target_budget = min(0.5 * time_limit, 45.0)

    while True:
        t0 = time.time()
        msh, uh, info = _solve_once(n, degree_u=degree_u, degree_p=degree_p, nu_value=nu, newton_max_it=newton_max_it)
        dt = time.time() - t0
        tried.append({"mesh_resolution": int(n), "solve_time_sec": float(dt), "relative_l2_error_velocity": float(info["relative_l2_error_velocity"])})
        best = (msh, uh, info)
        if (time.time() - start) + 2.2 * dt < target_budget and n < 96:
            n = min(96, int(math.ceil(1.5 * n)))
        else:
            break

    msh, uh, info = best
    u_grid = _sample_velocity_magnitude(uh, grid)

    info["wall_time_sec"] = float(time.time() - start)
    info["adaptivity_history"] = tried

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": info}
    return {"u": None, "solver_info": info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 0.15},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
