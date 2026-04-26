import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _build_case_defaults(case_spec: dict) -> dict:
    cs = dict(case_spec) if case_spec is not None else {}
    cs.setdefault("pde", {})
    cs["pde"].setdefault("time", {})
    cs["pde"]["time"].setdefault("t0", 0.0)
    cs["pde"]["time"].setdefault("t_end", 0.1)
    cs["pde"]["time"].setdefault("dt", 0.02)
    cs["pde"]["time"].setdefault("scheme", "backward_euler")
    cs.setdefault("output", {})
    cs["output"].setdefault("grid", {})
    cs["output"]["grid"].setdefault("nx", 64)
    cs["output"]["grid"].setdefault("ny", 64)
    cs["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return cs


def _probe_points_scalar(u_func: fem.Function, pts3: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    point_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, point_candidates, pts3)

    values = np.full((pts3.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    if domain.comm.size > 1:
        gathered = domain.comm.allgather(values)
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        values = out

    return values


def _sample_on_uniform_grid(u_func: fem.Function, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_points_scalar(u_func, pts3)
    return vals.reshape(ny, nx)


def _solve_heat(case_spec: dict, mesh_resolution: int, dt: float, degree: int = 1) -> Tuple[fem.Function, fem.Function, dict]:
    comm = MPI.COMM_WORLD
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.1))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    kappa_expr = 1.0 + 0.5 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f = fem.Constant(domain, ScalarType(1.0))

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) + np.cos(np.pi * X[1]))
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bc_dofs)

    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    u0.x.scatter_forward()

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa_expr * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    current_t = t0
    n_steps = math.ceil((t_end - t0) / dt)

    wall0 = time.perf_counter()
    for step in range(n_steps):
        target_t = min(t0 + (step + 1) * dt, t_end)
        actual_dt = target_t - current_t
        dt_c.value = ScalarType(actual_dt)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
        current_t = target_t
    wall1 = time.perf_counter()

    energy_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    energy_norm = comm.allreduce(energy_local, op=MPI.SUM)

    info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-9,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "wall_time_sec": wall1 - wall0,
        "verification": {
            "grad_L2_sq": float(energy_norm),
        },
    }
    return uh, u0, info


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    case_spec = _build_case_defaults(case_spec)
    comm = MPI.COMM_WORLD

    # Increased accuracy to better use time budget.
    mesh_resolution = 144
    degree = 1
    dt = 0.0025

    uh, u0, solver_info = _solve_heat(case_spec, mesh_resolution=mesh_resolution, dt=dt, degree=degree)

    # Accuracy verification through two-grid self-convergence on the required output grid.
    coarse_grid = dict(case_spec["output"]["grid"])
    check_case = {
        "pde": case_spec["pde"],
        "output": {"grid": coarse_grid},
    }
    uh_coarse, _, coarse_info = _solve_heat(check_case, mesh_resolution=96, dt=0.005, degree=degree)

    u_grid = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])
    u0_grid = _sample_on_uniform_grid(u0, case_spec["output"]["grid"])
    u_grid_coarse = _sample_on_uniform_grid(uh_coarse, case_spec["output"]["grid"])

    diff = u_grid - u_grid_coarse
    self_conv_l2 = float(np.sqrt(np.mean(diff**2)))
    self_conv_linf = float(np.max(np.abs(diff)))

    solver_info["verification"]["self_convergence_l2_grid"] = self_conv_l2
    solver_info["verification"]["self_convergence_linf_grid"] = self_conv_linf
    solver_info["verification"]["coarse_mesh_resolution"] = coarse_info["mesh_resolution"]
    solver_info["verification"]["coarse_dt"] = coarse_info["dt"]

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
