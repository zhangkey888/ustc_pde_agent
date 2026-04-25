import math
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_numpy(x: np.ndarray, t: float) -> np.ndarray:
    x0 = x[0]
    x1 = x[1]
    return np.exp(-t) * (
        np.sin(np.pi * x0) * np.sin(np.pi * x1)
        + 0.2 * np.sin(6.0 * np.pi * x0) * np.sin(6.0 * np.pi * x1)
    )


def _sample_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = points.T
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(points.shape[1], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(points.shape[1], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return np.empty(points.shape[1], dtype=np.float64)


def _sample_on_grid(u_func: fem.Function, grid: Dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_function(u_func, points)
    if u_func.function_space.mesh.comm.rank == 0:
        return vals.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _choose_resolution(case_spec: dict) -> Tuple[int, int, float]:
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))
    dt_suggested = float(pde_time.get("dt", 0.01))
    T = max(t_end - t0, 1.0e-12)

    mesh_resolution = 72
    degree = 2
    dt = min(dt_suggested, 0.0025)
    n_steps = max(1, int(math.ceil(T / dt)))
    dt = T / n_steps
    return mesh_resolution, degree, dt


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))

    mesh_resolution, degree, dt = _choose_resolution(case_spec)
    n_steps = int(round((t_end - t0) / dt))
    if n_steps < 1:
        n_steps = 1
        dt = t_end - t0 if t_end > t0 else 0.1

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    kappa = fem.Constant(msh, ScalarType(1.0))
    dt_c = fem.Constant(msh, ScalarType(dt))
    t_c = fem.Constant(msh, ScalarType(t0))

    def exact_ufl(tconst):
        return ufl.exp(-tconst) * (
            ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 0.2 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
        )

    u_exact_t = exact_ufl(t_c)
    f_expr = (-u_exact_t) - ufl.div(kappa * ufl.grad(u_exact_t))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)
    u0_fun = fem.Function(V)

    u0_fun.interpolate(lambda X: _exact_numpy(X, t0))
    u_n.x.array[:] = u0_fun.x.array
    u_n.x.scatter_forward()

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc.interpolate(lambda X: _exact_numpy(X, t0 + dt))
    bc = fem.dirichletbc(u_bc, bdofs)

    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=5000)

    total_iterations = 0

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        u_bc.interpolate(lambda X, tt=t_now: _exact_numpy(X, tt))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("Iterative solver did not converge")
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 1))
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(lambda X: _exact_numpy(X, t_end))
    err = fem.Function(V)
    err.x.array[:] = u_h.x.array - u_exact_final.x.array
    err.x.scatter_forward()
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_sq, op=MPI.SUM))

    u_grid = _sample_on_grid(u_h, case_spec["output"]["grid"])
    u_initial = _sample_on_grid(u0_fun, case_spec["output"]["grid"])

    if comm.rank == 0:
        return {
            "u": u_grid,
            "u_initial": u_initial,
            "solver_info": {
                "mesh_resolution": int(mesh_resolution),
                "element_degree": int(degree),
                "ksp_type": str(solver.getType()),
                "pc_type": str(solver.getPC().getType()),
                "rtol": float(1.0e-10),
                "iterations": int(total_iterations),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": "backward_euler",
                "l2_error": float(l2_error),
            },
        }
    return {"u": None, "u_initial": None, "solver_info": {}}
