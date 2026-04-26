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
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: hypre
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _get_case_params(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) or {}

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.3))
    dt_suggested = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "backward_euler")).lower()

    # Diffusion coefficient; default tuned for boundary-layer manufactured solution.
    epsilon = (
        pde.get("epsilon", None)
        if isinstance(pde, dict)
        else None
    )
    if epsilon is None:
        epsilon = case_spec.get("epsilon", None)
    if epsilon is None:
        epsilon = 0.01
    epsilon = float(epsilon)

    return {
        "t0": t0,
        "t_end": t_end,
        "dt": min(dt_suggested, 0.0025),
        "scheme": scheme,
        "epsilon": epsilon,
    }


def _choose_resolution(t_end: float, dt: float) -> tuple[int, int]:
    # Use available time budget proactively: second-order spatial accuracy with P2 and
    # a moderately refined mesh/time step. Chosen to remain well below 100s typically.
    # Increase resolution if time stepping count is small.
    n_steps = max(1, int(math.ceil((t_end) / dt)))
    if n_steps <= 60:
        return 128, 2
    if n_steps <= 120:
        return 112, 2
    return 64, 2


def _manufactured_exact_expr(msh, t_value: float):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t_value) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])


def _reaction_coeff(case_spec: dict) -> float:
    pde = case_spec.get("pde", {})
    # Default linear reaction R(u)=u unless specified otherwise.
    reaction = pde.get("reaction", None)
    if isinstance(reaction, dict):
        if reaction.get("type", "") == "linear":
            return float(reaction.get("coefficient", 1.0))
    if isinstance(pde, dict) and "reaction_coefficient" in pde:
        return float(pde["reaction_coefficient"])
    return 1.0


def _sample_on_grid(u_func: fem.Function, bbox, nx: int, ny: int) -> np.ndarray:
    msh = u_func.function_space.mesh
    xs = np.linspace(float(bbox[0]), float(bbox[1]), int(nx))
    ys = np.linspace(float(bbox[2]), float(bbox[3]), int(ny))
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

    values = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(idx_map, dtype=np.int32)] = vals

    if msh.comm.size > 1:
        recv = np.empty_like(values)
        msh.comm.Allreduce(values, recv, op=MPI.MAX)
        nan_mask = np.isnan(recv)
        if np.any(nan_mask):
            # If a point was not found on some procs, recover by taking non-NaN minimum/maximum logic.
            alt = np.where(np.isnan(values), -np.inf, values)
            recv2 = np.empty_like(alt)
            msh.comm.Allreduce(alt, recv2, op=MPI.MAX)
            recv[nan_mask] = recv2[nan_mask]
        values = recv

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    params = _get_case_params(case_spec)
    t0 = params["t0"]
    t_end = params["t_end"]
    dt = params["dt"]
    epsilon = params["epsilon"]
    reaction_coeff = _reaction_coeff(case_spec)

    # Accuracy-oriented defaults under the given runtime budget
    mesh_resolution, degree = _choose_resolution(t_end, dt)

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Boundary condition from exact solution
    u_bc = fem.Function(V)

    def interp_exact_time(t):
        def f(x):
            return np.exp(-t) * np.exp(4.0 * x[0]) * np.sin(np.pi * x[1])
        return f

    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc.interpolate(interp_exact_time(t0))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition from exact solution
    u_n = fem.Function(V)
    u_n.interpolate(interp_exact_time(t0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Manufactured exact solution and forcing for:
    # u_t - eps Δu + c u = f
    t_const = fem.Constant(msh, ScalarType(t0 + dt))
    u_exact_ufl = ufl.exp(-t_const) * ufl.exp(4.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = (
        -u_exact_ufl
        - epsilon * (16.0 - ufl.pi ** 2) * u_exact_ufl
        + reaction_coeff * u_exact_ufl
    )
    f_fun = fem.Function(V)
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)

    # Backward Euler
    a = (u * v + dt * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt * reaction_coeff * u * v) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    t = t0
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    # Store initial sampled field
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    u_initial_grid = _sample_on_grid(u_n, bbox, nx, ny)

    wall0 = time.perf_counter()

    for _ in range(n_steps):
        t = min(t + dt, t_end)
        t_const.value = ScalarType(t)
        u_bc.interpolate(interp_exact_time(t))
        f_fun.interpolate(f_expr)

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

    wall1 = time.perf_counter()

    # Accuracy verification in L2 and Linf against exact solution at final time
    u_ex_final = fem.Function(V)
    u_ex_final.interpolate(interp_exact_time(t_end))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_final.x.array
    err_fun.x.scatter_forward()

    l2_error_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))
    linf_local = np.max(np.abs(err_fun.x.array)) if err_fun.x.array.size > 0 else 0.0
    linf_error = comm.allreduce(linf_local, op=MPI.MAX)

    u_grid = _sample_on_grid(uh, bbox, nx, ny)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1.0e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error_verification": float(l2_error),
            "linf_error_verification": float(linf_error),
            "wall_time_sec_observed": float(wall1 - wall0),
            "epsilon": float(epsilon),
            "reaction_coefficient": float(reaction_coeff),
        },
        "u_initial": u_initial_grid,
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.005, "scheme": "backward_euler"},
            "epsilon": 0.01,
            "reaction_coefficient": 1.0,
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
