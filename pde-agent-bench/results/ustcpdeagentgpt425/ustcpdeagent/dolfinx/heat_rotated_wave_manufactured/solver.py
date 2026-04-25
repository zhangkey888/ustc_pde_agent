import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType


def _exact_u_numpy(x, y, t):
    return np.exp(-t) * np.sin(3.0 * np.pi * (x + y)) * np.sin(np.pi * (x - y))


def _forcing_ufl(x, t, kappa):
    u_exact = ufl.exp(-t) * ufl.sin(3.0 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    u_t = -u_exact
    lap_u = ufl.div(ufl.grad(u_exact))
    return u_t - kappa * lap_u


def _probe_function(u_func: fem.Function, points_array: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    pts = np.ascontiguousarray(points_array.T, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(points_array.shape[1], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)
        local_values[np.array(eval_map, dtype=np.int32)] = vals[:, 0]

    gathered = domain.comm.allgather(local_values)
    global_values = np.full(points_array.shape[1], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_values[mask] = arr[mask]

    if np.isnan(global_values).any():
        missing = np.where(np.isnan(global_values))[0]
        raise RuntimeError(f"Failed to evaluate solution at {len(missing)} grid points.")

    return global_values


def _sample_on_grid(u_func: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0 = float(case_spec.get("pde", {}).get("time", {}).get("t0", 0.0))
    t_end = float(case_spec.get("pde", {}).get("time", {}).get("t_end", 0.1))
    dt_in = case_spec.get("pde", {}).get("time", {}).get("dt", 0.01)

    if dt_in is None:
        dt_in = 0.01
    dt_in = float(dt_in)

    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))

    total_time = t_end - t0
    if total_time <= 0:
        raise ValueError("t_end must be greater than t0")

    # Accuracy-oriented defaults chosen to remain comfortably within the time budget.
    element_degree = 2
    mesh_resolution = 72
    target_steps = max(1, int(math.ceil(total_time / dt_in)))
    # Proactively refine temporal accuracy if suggested dt is coarse.
    target_steps = max(target_steps, 40)
    dt = total_time / target_steps
    n_steps = target_steps

    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)

    def u_exact_callable(t):
        return lambda X: np.exp(-t) * np.sin(3.0 * np.pi * (X[0] + X[1])) * np.sin(np.pi * (X[0] - X[1]))

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(u_exact_callable(t0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.name = "u_bc"

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    t_c = fem.Constant(domain, ScalarType(t0 + dt))
    f_expr = _forcing_ufl(x, t_c, kappa_c)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    uh.name = "u"

    grid_spec = case_spec["output"]["grid"]
    u_initial_grid = _sample_on_grid(u_n, grid_spec)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_c.value = ScalarType(t_now)
        u_bc.interpolate(u_exact_callable(t_now))
        u_bc.x.scatter_forward()

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

    elapsed = time.perf_counter() - start

    u_grid = _sample_on_grid(uh, grid_spec)

    # Accuracy verification against manufactured exact solution at final time.
    err_fn = fem.Function(V)
    err_fn.interpolate(u_exact_callable(t_end))
    err_form = fem.form((uh - err_fn) * (uh - err_fn) * ufl.dx)
    exact_form = fem.form(err_fn * err_fn * ufl.dx)
    l2_err_sq = fem.assemble_scalar(err_form)
    l2_exact_sq = fem.assemble_scalar(exact_form)
    l2_err_sq = comm.allreduce(l2_err_sq, op=MPI.SUM)
    l2_exact_sq = comm.allreduce(l2_exact_sq, op=MPI.SUM)
    l2_error = math.sqrt(max(l2_err_sq, 0.0))
    rel_l2_error = l2_error / max(math.sqrt(max(l2_exact_sq, 0.0)), 1e-16)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "wall_time_sec": float(elapsed),
        "l2_error": float(l2_error),
        "relative_l2_error": float(rel_l2_error),
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01},
            "coefficients": {"kappa": 1.0},
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
