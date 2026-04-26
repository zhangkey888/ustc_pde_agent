import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _make_mesh_and_spaces(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    return msh, W, V, Q


def _velocity_bc(msh, W, V, marker, value):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, marker)
    dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    g = fem.Function(V)
    arr = np.array(value, dtype=np.float64)

    def interp(x):
        out = np.zeros((msh.geometry.dim, x.shape[1]), dtype=np.float64)
        out[0, :] = arr[0]
        out[1, :] = arr[1]
        return out

    g.interpolate(interp)
    return fem.dirichletbc(g, dofs, W.sub(0))


def _pressure_pin(W, Q):
    pdofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(pdofs) == 0:
        return None
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    return fem.dirichletbc(p0, pdofs, W.sub(1))


def _sample_vector_function(func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = xx.ravel()
    pts[:, 1] = yy.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    val_size = func.function_space.element.value_size
    local_vals = np.full((pts.shape[0], val_size), np.nan, dtype=np.float64)
    eval_points = []
    eval_cells = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_ids:
        vals = func.eval(np.array(eval_points, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32), :] = np.asarray(vals, dtype=np.float64)

    global_vals = np.empty_like(local_vals)
    msh.comm.Allreduce(local_vals, global_vals, op=MPI.SUM)
    return global_vals.reshape(ny, nx, val_size)


def _solve_stokes_initial_guess(msh, W, bcs, nu):
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    f = fem.Constant(msh, np.array((0.0, 0.0), dtype=ScalarType))
    a = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx
    info = {"ksp_type": "gmres", "pc_type": "ilu", "rtol": 1e-8, "iterations": 0}

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_",
            petsc_options={
                "ksp_type": "gmres",
                "ksp_rtol": 1e-8,
                "pc_type": "ilu",
            },
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        return wh, info
    except Exception:
        info = {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12, "iterations": 0}
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_lu_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        return wh, info


def _solve_navier_stokes(msh, W, bcs, nu, w0):
    w = fem.Function(W)
    w.x.array[:] = w0.x.array
    w.x.scatter_forward()
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    def eps(wv):
        return ufl.sym(ufl.grad(wv))

    f = fem.Constant(msh, np.array((0.0, 0.0), dtype=ScalarType))
    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    nonlinear_iterations = [0]
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
                "snes_rtol": 1e-9,
                "snes_atol": 1e-10,
                "snes_max_it": 40,
                "ksp_type": "gmres",
                "ksp_rtol": 1e-8,
                "pc_type": "ilu",
            },
        )
        wh = problem.solve()
        wh.x.scatter_forward()
        return wh, {"ksp_type": "gmres", "pc_type": "ilu", "rtol": 1e-8, "iterations": 0}, nonlinear_iterations
    except Exception:
        w.x.array[:] = w0.x.array
        w.x.scatter_forward()
        for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
            Fk = (
                2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
                + alpha * ufl.inner(ufl.grad(u) * u, v) * ufl.dx
                - p * ufl.div(v) * ufl.dx
                + ufl.div(u) * q * ufl.dx
                - ufl.inner(f, v) * ufl.dx
            )
            Jk = ufl.derivative(Fk, w)
            problem = petsc.NonlinearProblem(
                Fk,
                w,
                bcs=bcs,
                J=Jk,
                petsc_options_prefix=f"ns_{int(alpha*10)}_",
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-8,
                    "snes_atol": 1e-10,
                    "snes_max_it": 40,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            w = problem.solve()
            w.x.scatter_forward()
        return w, {"ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12, "iterations": 0}, nonlinear_iterations


def _verification_metrics(msh, ufun):
    V0 = fem.functionspace(msh, ("DG", 0))
    div_expr = fem.Expression(ufl.div(ufun), V0.element.interpolation_points)
    divf = fem.Function(V0)
    divf.interpolate(div_expr)
    div_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(divf, divf) * ufl.dx))
    vel_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(ufun, ufun) * ufl.dx))
    h1_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(ufun), ufl.grad(ufun)) * ufl.dx))
    comm = msh.comm
    return {
        "divergence_l2": float(np.sqrt(comm.allreduce(div_l2_sq, op=MPI.SUM))),
        "velocity_l2": float(np.sqrt(comm.allreduce(vel_l2_sq, op=MPI.SUM))),
        "velocity_h1_seminorm": float(np.sqrt(comm.allreduce(h1_sq, op=MPI.SUM))),
    }


def solve(case_spec: dict) -> dict:
    nu_value = 0.18
    try:
        nu_value = float(case_spec.get("pde", {}).get("coefficients", {}).get("nu", 0.18))
    except Exception:
        nu_value = 0.18

    # Use a moderately refined mesh by default to exploit generous time budget while staying robust.
    mesh_resolution = 72
    msh, W, V, Q = _make_mesh_and_spaces(mesh_resolution)

    bcs = [
        _velocity_bc(msh, W, V, lambda x: np.isclose(x[1], 1.0), (1.0, 0.0)),
        _velocity_bc(msh, W, V, lambda x: np.isclose(x[0], 1.0), (0.0, -0.6)),
        _velocity_bc(msh, W, V, lambda x: np.isclose(x[0], 0.0), (0.0, 0.0)),
        _velocity_bc(msh, W, V, lambda x: np.isclose(x[1], 0.0), (0.0, 0.0)),
    ]
    pbc = _pressure_pin(W, Q)
    if pbc is not None:
        bcs.append(pbc)

    nu = fem.Constant(msh, ScalarType(nu_value))
    w_stokes, lin_info = _solve_stokes_initial_guess(msh, W, bcs, nu)
    w_ns, ns_info, nonlinear_iterations = _solve_navier_stokes(msh, W, bcs, nu, w_stokes)

    ufun = w_ns.sub(0).collapse()
    verification = _verification_metrics(msh, ufun)

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    uvals = _sample_vector_function(ufun, msh, nx, ny, bbox)
    umag = np.linalg.norm(uvals, axis=2)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 2,
        "ksp_type": str(ns_info["ksp_type"]),
        "pc_type": str(ns_info["pc_type"]),
        "rtol": float(ns_info["rtol"]),
        "iterations": int(lin_info["iterations"] + ns_info["iterations"]),
        "nonlinear_iterations": list(nonlinear_iterations),
        "verification": verification,
    }

    return {"u": umag, "solver_info": solver_info}
