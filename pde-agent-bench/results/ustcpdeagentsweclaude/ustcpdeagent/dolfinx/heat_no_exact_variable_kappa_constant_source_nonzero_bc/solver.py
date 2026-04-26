import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _get_nested(dct: dict, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _boundary_expr(x):
    return np.sin(np.pi * x[0]) + np.cos(np.pi * x[1])


def _kappa_expr(x):
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _make_bc_function(V):
    g = fem.Function(V)
    g.interpolate(_boundary_expr)
    return g


def _make_initial_function(V):
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    u0.x.scatter_forward()
    return u0


def _probe_function(u_func, points_xyz):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = np.asarray(points_xyz, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(idx_map, dtype=np.int32)] = vals
    return values


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)

    if np.isnan(vals).any():
        # robust fallback for boundary/corner points: clip slightly inside domain
        eps = 1e-12
        pts[:, 0] = np.clip(pts[:, 0], xmin + eps, xmax - eps)
        pts[:, 1] = np.clip(pts[:, 1], ymin + eps, ymax - eps)
        vals2 = _probe_function(u_func, pts)
        mask = np.isnan(vals)
        vals[mask] = vals2[mask]

    # final fallback for any remaining NaNs: boundary expression or zero
    if np.isnan(vals).any():
        mask = np.isnan(vals)
        vals[mask] = np.sin(np.pi * pts[mask, 0]) + np.cos(np.pi * pts[mask, 1])

    return vals.reshape(ny, nx)


def _run_heat(nx, degree, dt, t_end, ksp_type="cg", pc_type="hypre", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    g = _make_bc_function(V)
    bc = fem.dirichletbc(g, dofs)

    u_n = _make_initial_function(V)
    u = fem.Function(V)
    u.name = "u"

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    kappa = 1.0 + 0.5 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = fem.Constant(msh, ScalarType(1.0))
    dt_c = fem.Constant(msh, ScalarType(dt))

    a = (u_trial * v + dt_c * ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
    except Exception:
        pass

    total_iterations = 0
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()
        its = solver.getIterationNumber()
        if its is not None:
            total_iterations += int(its)
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    # Accuracy verification module:
    # 1) residual-like BC mismatch on boundary dofs
    bc_vals = _make_bc_function(V)
    bc_mismatch = 0.0
    if len(dofs) > 0:
        diff = u.x.array[dofs] - bc_vals.x.array[dofs]
        bc_mismatch = float(np.linalg.norm(diff, ord=np.inf))

    # 2) diagnostic positivity / magnitude summary
    local_min = float(np.min(u.x.array)) if u.x.array.size > 0 else 0.0
    local_max = float(np.max(u.x.array)) if u.x.array.size > 0 else 0.0
    sol_min = comm.allreduce(local_min, op=MPI.MIN)
    sol_max = comm.allreduce(local_max, op=MPI.MAX)

    diagnostics = {
        "boundary_bc_max_mismatch": bc_mismatch,
        "solution_min": float(sol_min),
        "solution_max": float(sol_max),
    }

    return msh, V, u, total_iterations, diagnostics


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    t0 = float(_get_nested(case_spec, ["pde", "time", "t0"], 0.0))
    t_end = float(_get_nested(case_spec, ["pde", "time", "t_end"], 0.1))
    dt_suggested = float(_get_nested(case_spec, ["pde", "time", "dt"], 0.02))
    scheme = _get_nested(case_spec, ["pde", "time", "scheme"], "backward_euler")
    grid_spec = case_spec["output"]["grid"]

    # Adaptive time-accuracy tradeoff:
    # choose finer dt than suggested and moderately fine mesh/degree to exploit budget
    degree = 1
    mesh_resolution = 72
    dt = min(dt_suggested, 0.01)
    n_steps = int(math.ceil((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt_suggested

    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9

    start = time.perf_counter()
    try:
        msh, V, u, iterations, diagnostics = _run_heat(
            mesh_resolution, degree, dt, t_end - t0,
            ksp_type=ksp_type, pc_type=pc_type, rtol=rtol
        )
    except Exception:
        # robust fallback
        ksp_type = "preonly"
        pc_type = "lu"
        msh, V, u, iterations, diagnostics = _run_heat(
            max(48, mesh_resolution // 2), degree, dt, t_end - t0,
            ksp_type=ksp_type, pc_type=pc_type, rtol=rtol
        )
        mesh_resolution = max(48, mesh_resolution // 2)

    elapsed = time.perf_counter() - start

    # Lightweight self-consistency verification if runtime leaves margin:
    verification = {"runtime_sec": float(elapsed)}
    if elapsed < 8.0:
        try:
            _, _, u_ref, _, _ = _run_heat(
                min(mesh_resolution + 8, 96), degree, dt / 2.0, t_end - t0,
                ksp_type="cg", pc_type="hypre", rtol=rtol
            )
            u_grid = _sample_on_grid(u, grid_spec)
            u_ref_grid = _sample_on_grid(u_ref, grid_spec)
            verification["self_consistency_linf"] = float(np.max(np.abs(u_ref_grid - u_grid)))
        except Exception:
            u_grid = _sample_on_grid(u, grid_spec)
            verification["self_consistency_linf"] = None
    else:
        u_grid = _sample_on_grid(u, grid_spec)
        verification["self_consistency_linf"] = None

    # Initial condition sampled on output grid
    init_grid = np.zeros_like(u_grid)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": str(scheme),
        "accuracy_verification": {**diagnostics, **verification},
    }

    return {
        "u": u_grid,
        "u_initial": init_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.02,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 32,
                "ny": 32,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
