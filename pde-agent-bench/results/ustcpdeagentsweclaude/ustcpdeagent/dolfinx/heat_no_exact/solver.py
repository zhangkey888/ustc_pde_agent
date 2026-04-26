DIAGNOSIS = r"""
equation_type: heat
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

METHOD = r"""
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: heat
"""

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


def _parse_case(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    coeffs = case_spec.get("coefficients", {})
    time_data = case_spec.get("time", case_spec.get("pde", {}).get("time", {}))
    output = case_spec.get("output", {})

    t0 = float(time_data.get("t0", 0.0))
    t_end = float(time_data.get("t_end", 0.1))
    dt_suggested = float(time_data.get("dt", 0.02))
    scheme = str(time_data.get("scheme", "backward_euler"))
    kappa = float(coeffs.get("kappa", 1.0))

    return {
        "t0": t0,
        "t_end": t_end,
        "dt_suggested": dt_suggested,
        "scheme": scheme,
        "kappa": kappa,
        "output": output,
    }


def _source_values(x):
    return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])


def _initial_values(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_boundary_condition(V, domain):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _probe_function(u_func, points_array):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    points_t = points_array.T
    cell_candidates = geometry.compute_collisions_points(tree, points_t)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_t)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_t[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full(points_array.shape[1], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_vals)
    result = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(result) & ~np.isnan(arr)
        result[mask] = arr[mask]
    return np.nan_to_num(result, nan=0.0)


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _compute_accuracy_metrics(domain, uh, u_prev, dt, kappa):
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    energy = fem.assemble_scalar(fem.form(0.5 * uh * uh * ufl.dx))
    grad_energy = fem.assemble_scalar(fem.form(kappa * ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    l2_increment_sq = fem.assemble_scalar(fem.form((uh - u_prev) * (uh - u_prev) * ufl.dx))
    forcing_work = fem.assemble_scalar(fem.form(f_expr * uh * ufl.dx))

    comm = domain.comm
    energy = comm.allreduce(energy, op=MPI.SUM)
    grad_energy = comm.allreduce(grad_energy, op=MPI.SUM)
    l2_increment_sq = comm.allreduce(l2_increment_sq, op=MPI.SUM)
    forcing_work = comm.allreduce(forcing_work, op=MPI.SUM)

    temporal_increment_l2 = math.sqrt(max(l2_increment_sq, 0.0))
    residual_indicator = temporal_increment_l2 / max(dt, 1e-14)

    return {
        "energy": float(energy),
        "grad_energy": float(grad_energy),
        "temporal_increment_l2": float(temporal_increment_l2),
        "forcing_work": float(forcing_work),
        "residual_indicator": float(residual_indicator),
    }


def solve(case_spec: dict) -> dict:
    params = _parse_case(case_spec)
    comm = MPI.COMM_WORLD

    t0 = params["t0"]
    t_end = params["t_end"]
    dt_suggested = params["dt_suggested"]
    scheme = params["scheme"]
    kappa_value = params["kappa"]

    interval = max(t_end - t0, 1e-14)

    mesh_resolution = 56
    element_degree = 1
    target_dt = min(dt_suggested, 0.01)
    n_steps = max(8, int(math.ceil(interval / target_dt)))
    dt = interval / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    bc = _build_boundary_condition(V, domain)

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(_initial_values)
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    kappa = fem.Constant(domain, ScalarType(kappa_value))
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
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
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=5000)

    uh = fem.Function(V)
    uh.name = "u"

    total_iterations = 0
    metrics = []
    wall_t0 = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())
        metrics.append(_compute_accuracy_metrics(domain, uh, u_n, dt, kappa_value))
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall_time = time.perf_counter() - wall_t0

    output_grid = params["output"]["grid"]
    u_grid = _sample_on_uniform_grid(uh, output_grid)

    u_initial_fun = fem.Function(V)
    u_initial_fun.interpolate(_initial_values)
    u_initial_fun.x.scatter_forward()
    u_initial_grid = _sample_on_uniform_grid(u_initial_fun, output_grid)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(1e-9),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": str(scheme),
        "wall_time_observed": float(wall_time),
        "accuracy_verification": {
            "final_energy": metrics[-1]["energy"] if metrics else 0.0,
            "final_grad_energy": metrics[-1]["grad_energy"] if metrics else 0.0,
            "final_temporal_increment_l2": metrics[-1]["temporal_increment_l2"] if metrics else 0.0,
            "final_residual_indicator": metrics[-1]["residual_indicator"] if metrics else 0.0,
        },
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "coefficients": {"kappa": 1.0},
        "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 21, "ny": 17, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": True},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
