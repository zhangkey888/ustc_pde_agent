import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: nonlinear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: localized_source
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _get_grid(case_spec: Dict[str, Any]) -> Tuple[int, int, Tuple[float, float, float, float]]:
    grid = case_spec.get("output", {}).get("grid", {})
    nx = int(grid.get("nx", 64))
    ny = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _get_time(case_spec: Dict[str, Any]) -> Tuple[float, float, float]:
    t = case_spec.get("pde", {}).get("time", {})
    t0 = float(t.get("t0", 0.0))
    t_end = float(t.get("t_end", 0.35))
    dt_suggested = float(t.get("dt", 0.01))
    return t0, t_end, dt_suggested


def _choose_params(case_spec: Dict[str, Any], nx_out: int, ny_out: int) -> Tuple[int, int, float]:
    t0, t_end, dt_suggested = _get_time(case_spec)
    mesh_n = int(case_spec.get("mesh_resolution", max(80, min(128, max(nx_out, ny_out)))))
    degree = int(case_spec.get("element_degree", 1))
    dt = min(dt_suggested, max((t_end - t0) / 120.0, 0.0025))
    return mesh_n, degree, dt


def _u0_fun(x):
    return 0.4 + 0.1 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _make_bc(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _sample(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(bb, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local = np.full(nx * ny, np.nan, dtype=np.float64)
    eval_pts = []
    eval_cells = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            ids.append(i)

    if eval_pts:
        vals = uh.eval(np.asarray(eval_pts, dtype=np.float64), np.asarray(eval_cells, dtype=np.int32))
        local[np.asarray(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = COMM.gather(local, root=0)
    if COMM.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            m = ~np.isnan(arr)
            out[m] = arr[m]
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    nx_out, ny_out, bbox = _get_grid(case_spec)
    t0, t_end, _ = _get_time(case_spec)
    mesh_n, degree, dt = _choose_params(case_spec, nx_out, ny_out)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / max(n_steps, 1)

    domain = mesh.create_unit_square(COMM, mesh_n, mesh_n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    bc = _make_bc(V, domain)

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(float(case_spec.get("epsilon", 0.01)))
    rho = ScalarType(float(case_spec.get("reaction_rho", 2.0)))

    f_expr = 4.0 * ufl.exp(-200.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2)) - 2.0 * ufl.exp(
        -200.0 * ((x[0] - 0.65) ** 2 + (x[1] - 0.35) ** 2)
    )

    u_n = fem.Function(V)
    u_n.interpolate(_u0_fun)
    u_n.x.scatter_forward()

    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    u = fem.Function(V)
    v = ufl.TestFunction(V)

    reaction = rho * u * (1.0 - u)
    F = ((u - u_n) / dt) * v * ufl.dx + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction * v * ufl.dx - f_expr * v * ufl.dx
    J = ufl.derivative(F, u)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8

    nonlinear_iterations = []
    linear_iterations_total = 0

    t_start = time.perf_counter()
    for _ in range(n_steps):
        u.x.array[:] = u_n.x.array
        u.x.scatter_forward()
        problem = petsc.NonlinearProblem(
            F,
            u,
            bcs=[bc],
            J=J,
            petsc_options_prefix="rd_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-8,
                "snes_atol": 1e-10,
                "snes_max_it": 20,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
            },
        )
        u = problem.solve()
        u.x.scatter_forward()
        try:
            snes = problem.solver
            nonlinear_iterations.append(int(snes.getIterationNumber()))
            linear_iterations_total += int(snes.getLinearSolveIterations())
        except Exception:
            nonlinear_iterations.append(-1)
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()
    elapsed = time.perf_counter() - t_start

    res_form = fem.form(F)
    res_vec = petsc.create_vector(res_form.function_spaces)
    with res_vec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(res_vec, res_form)
    res_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    residual_norm = float(res_vec.norm())

    local_min = np.min(u_n.x.array) if len(u_n.x.array) else 0.0
    local_max = np.max(u_n.x.array) if len(u_n.x.array) else 0.0
    umin = COMM.allreduce(float(local_min), op=MPI.MIN)
    umax = COMM.allreduce(float(local_max), op=MPI.MAX)

    u_grid = _sample(domain, u_n, nx_out, ny_out, bbox)
    u0_grid = _sample(domain, u_initial, nx_out, ny_out, bbox)

    result = None
    if COMM.rank == 0:
        result = {
            "u": u_grid,
            "u_initial": u0_grid,
            "solver_info": {
                "mesh_resolution": mesh_n,
                "element_degree": degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": int(linear_iterations_total),
                "dt": float(dt),
                "n_steps": int(n_steps),
                "time_scheme": "backward_euler",
                "nonlinear_iterations": [int(it) for it in nonlinear_iterations],
                "verification": {
                    "residual_norm": residual_norm,
                    "u_min": umin,
                    "u_max": umax,
                    "wall_time_sec": elapsed,
                },
            },
        }
    return COMM.bcast(result, root=0)


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.35, "dt": 0.01, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
