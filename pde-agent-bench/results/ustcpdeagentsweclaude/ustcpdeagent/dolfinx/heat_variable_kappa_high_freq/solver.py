import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _exact_u_expr(x, t):
    return np.exp(-t) * np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _kappa_numpy(x):
    return 1.0 + 0.3 * np.sin(6.0 * np.pi * x[0]) * np.sin(6.0 * np.pi * x[1])


def _sample_function_on_grid(domain, uh: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals.real

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            missing = np.isnan(global_vals).sum()
            raise RuntimeError(f"Failed to evaluate {missing} grid points.")
        return global_vals.reshape((ny, nx))
    else:
        return np.empty((ny, nx), dtype=np.float64)


def _run_simulation(mesh_resolution: int, degree: int, dt: float, t_end: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    tdim = domain.topology.dim
    fdim = tdim - 1

    t_const = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])

    grad_u_exact = ufl.grad(u_exact_ufl)
    div_term = ufl.div(kappa * grad_u_exact)
    f_ufl = -u_exact_ufl - div_term

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _exact_u_expr(X, 0.0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_const * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)

    n_steps = int(round(t_end / dt))
    total_iters = 0

    t0_wall = time.perf_counter()

    for n in range(1, n_steps + 1):
        t_now = n * dt
        t_const.value = ScalarType(t_now)
        u_bc.interpolate(lambda X, tt=t_now: _exact_u_expr(X, tt))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - t0_wall

    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(lambda X, tt=t_end: _exact_u_expr(X, tt))
    u_exact_final.x.scatter_forward()

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_final.x.array
    err_fun.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "u_final": uh,
        "u_initial_array": None,
        "l2_error": l2_error,
        "wall_time": wall,
        "iterations": int(total_iters),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", 0.0))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", 0.1))
    dt_suggested = float(case_spec.get("pde", {}).get("time", {}).get("dt", 0.005))
    output_grid = case_spec["output"]["grid"]

    candidates = [
        (48, 1, min(dt_suggested, 0.0025)),
        (64, 1, min(dt_suggested, 0.0020)),
        (48, 2, min(dt_suggested, 0.0025)),
        (64, 2, min(dt_suggested, 0.0020)),
    ]

    chosen = None
    best = None
    time_budget = 21.0

    for mesh_resolution, degree, dt in candidates:
        dt_eff = (t_end - t0) / int(round((t_end - t0) / dt))
        result = _run_simulation(mesh_resolution, degree, dt_eff, t_end - t0)
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result
        if result["l2_error"] <= 3.74e-3:
            chosen = result
            if result["wall_time"] > 0.35 * time_budget:
                break

    if chosen is None:
        chosen = best

    if chosen["wall_time"] < 4.0:
        refined_candidates = []
        if chosen["element_degree"] == 1:
            refined_candidates.append((max(chosen["mesh_resolution"], 64), 2, max(chosen["dt"] / 2.0, 0.001)))
        refined_candidates.append((chosen["mesh_resolution"] + 16, chosen["element_degree"], max(chosen["dt"] / 2.0, 0.001)))
        refined_candidates.append((chosen["mesh_resolution"] + 32, 2, max(chosen["dt"] / 2.0, 0.001)))
        for mesh_resolution, degree, dt in refined_candidates:
            nsteps = max(1, int(round((t_end - t0) / dt)))
            dt_eff = (t_end - t0) / nsteps
            trial = _run_simulation(mesh_resolution, degree, dt_eff, t_end - t0)
            if trial["wall_time"] <= time_budget and trial["l2_error"] <= chosen["l2_error"]:
                chosen = trial
            if trial["wall_time"] > 0.7 * time_budget:
                break

    sampled_final = _sample_function_on_grid(chosen["domain"], chosen["u_final"], output_grid)

    initial_grid = None
    if comm.rank == 0:
        xs = np.linspace(output_grid["bbox"][0], output_grid["bbox"][1], int(output_grid["nx"]))
        ys = np.linspace(output_grid["bbox"][2], output_grid["bbox"][3], int(output_grid["ny"]))
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        initial_grid = (
            np.sin(2.0 * np.pi * XX) * np.sin(2.0 * np.pi * YY)
        ).astype(np.float64)

    solver_info = {
        "mesh_resolution": chosen["mesh_resolution"],
        "element_degree": chosen["element_degree"],
        "ksp_type": chosen["ksp_type"],
        "pc_type": chosen["pc_type"],
        "rtol": chosen["rtol"],
        "iterations": chosen["iterations"],
        "dt": chosen["dt"],
        "n_steps": chosen["n_steps"],
        "time_scheme": "backward_euler",
        "l2_error_verification": chosen["l2_error"],
        "wall_time_verification": chosen["wall_time"],
    }

    if comm.rank == 0:
        return {"u": sampled_final, "u_initial": initial_grid, "solver_info": solver_info}
    else:
        return {"u": sampled_final, "u_initial": np.empty_like(sampled_final), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005}
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
