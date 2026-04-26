import math
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get(case_spec: dict, path, default):
    cur = case_spec
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _sample_function(u_func: fem.Function, points_3xn: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    pts = points_3xn.T.copy()
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(points_3xn.shape[1], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(points_3xn.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _sample_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD

    t0 = float(_get(case_spec, ["pde", "time", "t0"], 0.0))
    t_end = float(_get(case_spec, ["pde", "time", "t_end"], 0.08))
    dt_in = float(_get(case_spec, ["pde", "time", "dt"], 0.008))
    output_grid = _get(case_spec, ["output", "grid"], {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})

    # Accuracy/time tradeoff tuned for this manufactured boundary-layer case
    mesh_resolution = 64
    element_degree = 2
    dt = min(dt_in, 0.004)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps
    kappa_val = 1.0

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    t_const = fem.Constant(domain, ScalarType(t0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa = fem.Constant(domain, ScalarType(kappa_val))

    def u_exact_ufl(t_symbol):
        return ufl.exp(-t_symbol) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])

    def f_ufl(t_symbol):
        uex = u_exact_ufl(t_symbol)
        u_t = -uex
        lap_uex = (25.0 - ufl.pi ** 2) * uex
        return u_t - kappa * lap_uex

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_bc = fem.Function(V)

    u_n.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))
    u_h.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))
    u_bc.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_ufl(t_const) * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=5000)

    total_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        u_bc.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, u_h.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("KSP failed")
        except Exception:
            solver = PETSc.KSP().create(comm)
            solver.setOperators(A)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.solve(b, u_h.x.petsc_vec)

        u_h.x.scatter_forward()
        total_iterations += max(0, solver.getIterationNumber())
        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = u_h.x.array - u_ex.x.array
    e.x.scatter_forward()

    l2_sq_local = fem.assemble_scalar(fem.form(e * e * ufl.dx))
    l2_sq = comm.allreduce(l2_sq_local, op=MPI.SUM)
    l2_error = math.sqrt(l2_sq)

    u_grid = _sample_on_grid(u_h, output_grid)
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    xmin, xmax, ymin, ymax = map(float, output_grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_initial = np.exp(-t0) * np.exp(5.0 * YY) * np.sin(np.pi * XX)

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
        },
    }
