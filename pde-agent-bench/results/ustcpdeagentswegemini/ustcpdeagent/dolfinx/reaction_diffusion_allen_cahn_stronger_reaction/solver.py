import math
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
# preconditioner: ilu
# special_treatment: problem_splitting
# pde_skill: reaction_diffusion
# ```


ScalarType = PETSc.ScalarType


def _extract_time(case_spec: dict) -> Tuple[float, float, float]:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", case_spec.get("t_end", 0.1)))
    dt = float(time_spec.get("dt", case_spec.get("dt", 0.002)))
    if dt <= 0:
        dt = 0.002
    if t_end <= t0:
        t_end = t0 + dt
    return t0, t_end, dt


def _make_exact_and_forcing(msh, epsilon: float, reaction_lambda: float, t_symbol):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.exp(-t_symbol) * (
        ScalarType(0.15) + ScalarType(0.12) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    )
    f_expr = -u_exact - epsilon * ufl.div(ufl.grad(u_exact)) + reaction_lambda * (u_exact**3 - u_exact)
    return u_exact, f_expr


def _sample_function_on_grid(u_fun: fem.Function, msh, grid_spec: dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.asarray(points_on_proc, dtype=np.float64),
                          np.asarray(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(ids, dtype=np.int32)] = vals

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Some sampling points were not evaluated.")
        return out.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _compute_rel_l2_error(msh, uh: fem.Function, u_exact_ufl) -> float:
    err_sq = fem.assemble_scalar(fem.form(((uh - u_exact_ufl) ** 2) * ufl.dx))
    ref_sq = fem.assemble_scalar(fem.form((u_exact_ufl ** 2) * ufl.dx))
    err_sq = msh.comm.allreduce(err_sq, op=MPI.SUM)
    ref_sq = msh.comm.allreduce(ref_sq, op=MPI.SUM)
    if ref_sq <= 0.0:
        return float(math.sqrt(max(err_sq, 0.0)))
    return float(math.sqrt(max(err_sq, 0.0) / ref_sq))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0, t_end, dt_suggested = _extract_time(case_spec)

    params = case_spec.get("params", {})
    epsilon = float(case_spec.get("epsilon", params.get("epsilon", 0.01)))
    reaction_lambda = float(case_spec.get("reaction_lambda", params.get("reaction_lambda", 12.0)))

    # Faster defaults, but still accurate for smooth manufactured data
    mesh_resolution = int(case_spec.get("mesh_resolution", params.get("mesh_resolution", 44)))
    degree = int(case_spec.get("element_degree", params.get("element_degree", 1)))
    dt = float(case_spec.get("dt", params.get("dt", min(dt_suggested, 0.002))))
    if dt <= 0:
        dt = min(dt_suggested, 0.002)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    t_const = fem.Constant(msh, ScalarType(t0))
    dt_const = fem.Constant(msh, ScalarType(dt))

    u_exact, f_expr = _make_exact_and_forcing(msh, epsilon, reaction_lambda, t_const)

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    u_n.x.scatter_forward()

    output_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]})
    u_initial = _sample_function_on_grid(u_n, msh, output_grid) if comm.rank == 0 else None

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Semi-implicit linearization:
    # (u^{n+1}-u^n)/dt - eps Δu^{n+1} + lambda((u^n)^2 u^{n+1} - u^{n+1}) = f^{n+1}
    a = (
        (1.0 / dt_const) * u * v * ufl.dx
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + reaction_lambda * ((u_n * u_n - 1.0) * u) * v * ufl.dx
    )
    L = ((1.0 / dt_const) * u_n + f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-9)
    ksp.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    nonlinear_iterations = []
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_const.value = ScalarType(t0 + step * dt)

        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        u_bc.x.scatter_forward()

        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        total_iterations += int(ksp.getIterationNumber())
        nonlinear_iterations.append(1)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start
    rel_l2_error = _compute_rel_l2_error(msh, uh, u_exact)
    u_grid = _sample_function_on_grid(uh, msh, output_grid) if comm.rank == 0 else None

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": nonlinear_iterations,
        "epsilon": float(epsilon),
        "reaction_lambda": float(reaction_lambda),
        "l2_rel_error": float(rel_l2_error),
        "wall_time_sec": float(elapsed),
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}
    return {
        "u": np.empty((output_grid["ny"], output_grid["nx"]), dtype=np.float64),
        "u_initial": np.empty((output_grid["ny"], output_grid["nx"]), dtype=np.float64),
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.002}},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
