import math
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _with_defaults(case_spec: dict) -> dict:
    case_spec = dict(case_spec or {})
    case_spec.setdefault("pde", {})
    case_spec["pde"].setdefault("time", {})
    case_spec["pde"]["time"].setdefault("t0", 0.0)
    case_spec["pde"]["time"].setdefault("t_end", 0.2)
    case_spec["pde"]["time"].setdefault("dt", 0.02)
    case_spec["pde"]["time"].setdefault("scheme", "backward_euler")
    case_spec.setdefault("output", {})
    case_spec["output"].setdefault("grid", {})
    case_spec["output"]["grid"].setdefault("nx", 64)
    case_spec["output"]["grid"].setdefault("ny", 64)
    case_spec["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case_spec


def _sample_on_grid(domain, ufun: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    eval_pts = []
    eval_cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if eval_pts:
        vals = ufun.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(eval_pts), -1)[:, 0]
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
    else:
        merged = None
    merged = comm.bcast(merged, root=0)
    return merged.reshape((ny, nx))


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    case_spec = _with_defaults(case_spec)
    comm = MPI.COMM_WORLD

    t0 = float(case_spec["pde"]["time"]["t0"])
    t_end = float(case_spec["pde"]["time"]["t_end"])
    dt_user = float(case_spec["pde"]["time"]["dt"])

    # Higher-than-requested temporal accuracy because the time budget is generous.
    dt = min(dt_user, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    mesh_resolution = 80
    element_degree = 1

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    eps = 0.05
    beta_np = np.array([2.0, 1.0], dtype=np.float64)
    beta = fem.Constant(domain, np.array(beta_np, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(eps))
    dt_c = fem.Constant(domain, ScalarType(dt))
    t_c = fem.Constant(domain, ScalarType(t0))

    pi = math.pi

    def u_exact_expr(tt):
        return ufl.exp(-2.0 * tt) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    u_exact_t = u_exact_expr(t_c)
    f_expr = (
        -2.0 * u_exact_t
        - eps_c * ufl.div(ufl.grad(u_exact_t))
        + ufl.dot(beta, ufl.grad(u_exact_t))
    )

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-2.0 * t0) * np.sin(pi * X[0]) * np.sin(pi * X[1]))

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(-2.0 * t0) * np.sin(pi * X[0]) * np.sin(pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = float(np.linalg.norm(beta_np))
    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (4.0 * eps / h**2) ** 2)

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + f_expr * v * ufl.dx
        + tau * ((u_n / dt_c) + f_expr) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        u_bc.interpolate(lambda X, tt=t: np.exp(-2.0 * tt) * np.sin(pi * X[0]) * np.sin(pi * X[1]))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("iterative solve failed")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(max(1, solver.getIterationNumber()))
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u0_fun = fem.Function(V)
    u0_fun.interpolate(lambda X: np.exp(-2.0 * t0) * np.sin(pi * X[0]) * np.sin(pi * X[1]))

    uex_final = fem.Function(V)
    uex_final.interpolate(lambda X: np.exp(-2.0 * t_end) * np.sin(pi * X[0]) * np.sin(pi * X[1]))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - uex_final.x.array
    err.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(domain, uh, grid)
    u0_grid = _sample_on_grid(domain, u0_fun, grid)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1e-9,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
        },
    }
