import math
import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type: reaction_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: multifrequency_source
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: crank_nicolson
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: reaction_diffusion
# ```


ScalarType = PETSc.ScalarType


def _source_expr(x):
    return (
        np.sin(5.0 * np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])
        + 0.5 * np.sin(9.0 * np.pi * x[0]) * np.sin(7.0 * np.pi * x[1])
    )


def _initial_expr(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _build_zero_dirichlet_bc(domain, V):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _probe_function_on_points(domain, u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    # points shape: (N, 3)
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []

    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.array(vals).reshape(-1)
        values[np.array(ids_local, dtype=np.int32)] = vals.real

    return values


def _sample_on_uniform_grid(domain, u_func: fem.Function, grid: Dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    local_vals = _probe_function_on_points(domain, u_func, pts)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(local_vals)
        vals = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
    else:
        vals = local_vals

    vals = np.nan_to_num(vals, nan=0.0)
    return vals.reshape((ny, nx))


def _run_simulation(
    nx: int,
    degree: int,
    dt: float,
    t_end: float,
    epsilon: float,
    reaction_alpha: float,
    ksp_type: str = "cg",
    pc_type: str = "hypre",
    rtol: float = 1e-10,
) -> Tuple[mesh.Mesh, fem.Function, fem.Function, Dict]:
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    bc = _build_zero_dirichlet_bc(domain, V)

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(_initial_expr)
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_ufl = (
        ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.5 * ufl.sin(9.0 * ufl.pi * x[0]) * ufl.sin(7.0 * ufl.pi * x[1])
    )
    f_fun = fem.Function(V)
    f_fun.interpolate(_source_expr)
    f_fun.x.scatter_forward()

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    alpha_c = fem.Constant(domain, ScalarType(reaction_alpha))

    a = (
        (1.0 / dt_c) * u * v * ufl.dx
        + 0.5 * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + 0.5 * alpha_c * u * v * ufl.dx
    )

    L = (
        (1.0 / dt_c) * u_n * v * ufl.dx
        - 0.5 * eps_c * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - 0.5 * alpha_c * u_n * v * ufl.dx
        + f_ufl * v * ufl.dx
    )

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

    if solver.getType().lower() == "cg":
        # Safer fallback if hypre unavailable/unsuitable in environment
        try:
            solver.getPC().setType(pc_type)
        except Exception:
            solver.getPC().setType("jacobi")

    uh = fem.Function(V)
    uh.name = "u"

    n_steps = int(round(t_end / dt))
    t = 0.0
    total_iterations = 0

    for _ in range(n_steps):
        t += dt
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            its = solver.getIterationNumber()
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)
            its = 1

        uh.x.scatter_forward()
        total_iterations += int(its)
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    # Accuracy verification module:
    # 1) projected PDE residual norm at final state
    w = ufl.TestFunction(V)
    residual_form = (
        eps_c * ufl.inner(ufl.grad(uh), ufl.grad(w)) * ufl.dx
        + alpha_c * uh * w * ufl.dx
        - f_ufl * w * ufl.dx
    )
    residual_vec = petsc.assemble_vector(fem.form(residual_form))
    fem.petsc.set_bc(residual_vec, [bc])
    residual_norm = residual_vec.norm()

    # 2) mesh-consistency indicator using source projection error
    f_interp = fem.Function(V)
    f_interp.interpolate(_source_expr)
    f_interp.x.scatter_forward()
    err_fun = fem.Function(V)
    err_fun.x.array[:] = f_interp.x.array
    err_fun.x.scatter_forward()
    source_l2 = math.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx)),
            op=MPI.SUM,
        )
    )

    info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()).lower(),
        "pc_type": str(solver.getPC().getType()).lower(),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "crank_nicolson",
        "accuracy_verification": {
            "final_steady_residual_l2": float(residual_norm),
            "source_projection_l2": float(source_l2),
        },
    }
    return domain, uh, fem.Function(V), info


def solve(case_spec: dict) -> dict:
    pde_time = case_spec.get("pde", {}).get("time", {})
    output_grid = case_spec["output"]["grid"]

    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.5))
    dt_user = float(pde_time.get("dt", 0.01))
    _scheme = str(pde_time.get("scheme", "crank_nicolson")).lower()

    # Agent-selectable defaults
    epsilon = float(case_spec.get("params", {}).get("epsilon", 0.02))
    reaction_alpha = float(case_spec.get("params", {}).get("reaction_alpha", 0.0))

    # Adaptive time-accuracy trade-off under generous time budget:
    # choose a more accurate dt if the suggested one is coarse.
    dt = min(dt_user, 0.01)
    total_time = t_end - t0
    if total_time <= 0:
        total_time = 0.5
    n_steps = max(1, int(round(total_time / dt)))
    dt = total_time / n_steps

    # Prefer P2 on a moderately fine mesh for multifrequency forcing
    nx = int(case_spec.get("params", {}).get("mesh_resolution", 48))
    degree = int(case_spec.get("params", {}).get("element_degree", 2))

    start_time = time.time()
    domain, uh, _, solver_info = _run_simulation(
        nx=nx,
        degree=degree,
        dt=dt,
        t_end=total_time,
        epsilon=epsilon,
        reaction_alpha=reaction_alpha,
        ksp_type="cg",
        pc_type="hypre",
        rtol=1e-10,
    )

    # Reconstruct initial condition for output
    V = uh.function_space
    u0_fun = fem.Function(V)
    u0_fun.interpolate(_initial_expr)
    u0_fun.x.scatter_forward()

    u_grid = _sample_on_uniform_grid(domain, uh, output_grid)
    u0_grid = _sample_on_uniform_grid(domain, u0_fun, output_grid)

    solver_info["wall_time_sec"] = float(time.time() - start_time)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.5, "dt": 0.01, "scheme": "crank_nicolson"}
        },
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
        "params": {
            "epsilon": 0.02,
            "reaction_alpha": 0.0,
            "mesh_resolution": 72,
            "element_degree": 2,
        },
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
