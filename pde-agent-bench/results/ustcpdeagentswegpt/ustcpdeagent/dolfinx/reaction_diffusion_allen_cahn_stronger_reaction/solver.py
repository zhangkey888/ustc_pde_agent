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
# nonlinear_solver: newton
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: reaction_diffusion
# ```

import time
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _extract_time(case_spec):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt = float(time_spec.get("dt", 0.002))
    if dt <= 0:
        dt = 0.002
    return t0, t_end, dt


def _extract_params(case_spec):
    pde = case_spec.get("pde", {})
    epsilon = float(pde.get("epsilon", case_spec.get("epsilon", 0.01)))
    reaction_lambda = float(pde.get("reaction_lambda", case_spec.get("reaction_lambda", 10.0)))
    return epsilon, reaction_lambda


def _choose_discretization(case_spec):
    time_limit = float(case_spec.get("time_limit_sec", 137.916))
    if time_limit > 90.0:
        return 96, 2, 0.001
    if time_limit > 40.0:
        return 80, 2, 0.00125
    return 64, 2, 0.002


def _gather_grid_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cands, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx * ny):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            out[np.isnan(out)] = 0.0
        return out.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t_wall0 = time.perf_counter()

    t0, t_end, dt_in = _extract_time(case_spec)
    epsilon, reaction_lambda = _extract_params(case_spec)
    mesh_resolution, element_degree, dt_target = _choose_discretization(case_spec)
    dt = min(dt_in, dt_target)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps if t_end > t0 else dt

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    t_c = fem.Constant(msh, ScalarType(t0))
    dt_c = fem.Constant(msh, ScalarType(dt))

    def u_exact_expr(t):
        return ufl.exp(-t) * (ScalarType(0.15) + ScalarType(0.12) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1]))

    def reaction(u):
        return reaction_lambda * (u**3 - u)

    def source_expr(t):
        ue = u_exact_expr(t)
        ue_t = -ue
        lap_ue = ufl.div(ufl.grad(ue))
        return ue_t - epsilon * lap_ue + reaction(ue)

    u_n = fem.Function(V)
    u = fem.Function(V)
    u.interpolate(fem.Expression(u_exact_expr(t_c), V.element.interpolation_points))
    u_n.x.array[:] = u.x.array

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr(t_c), V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    v = ufl.TestFunction(V)
    F = ((u - u_n) / dt_c) * v * ufl.dx + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + reaction(u) * v * ufl.dx - source_expr(t_c) * v * ufl.dx
    J = ufl.derivative(F, u)

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-9

    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    snes = problem.solver
    ksp = snes.getKSP()

    output_grid = case_spec.get("output", {}).get("grid", {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]})
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    u_initial = _gather_grid_function(u_n, bbox, nx, ny)

    nonlinear_iterations = []
    total_linear_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        u_bc.interpolate(fem.Expression(u_exact_expr(t_c), V.element.interpolation_points))
        u.x.array[:] = u_n.x.array
        try:
            snes.solve(None, u.x.petsc_vec)
        except Exception:
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp_type = "preonly"
            pc_type = "lu"
            snes.solve(None, u.x.petsc_vec)
        u.x.scatter_forward()
        nonlinear_iterations.append(int(snes.getIterationNumber()))
        total_linear_iterations += int(ksp.getTotalIterations())
        u_n.x.array[:] = u.x.array
        u_n.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr(t_c), V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = u.x.array - u_exact.x.array
    e.x.scatter_forward()
    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = float(np.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM)))

    u_grid = _gather_grid_function(u, bbox, nx, ny)
    exact_grid = _gather_grid_function(u_exact, bbox, nx, ny) if rank == 0 else np.empty((ny, nx))
    grid_linf = float(np.max(np.abs(u_grid - exact_grid))) if rank == 0 else 0.0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(total_linear_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [int(v) for v in nonlinear_iterations],
        "l2_error": float(l2_error),
        "grid_linf_error": float(grid_linf),
        "wall_time_sec": float(time.perf_counter() - t_wall0),
        "epsilon": float(epsilon),
        "reaction_lambda": float(reaction_lambda),
    }

    if rank == 0:
        return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}
    return {"u": np.empty((ny, nx)), "u_initial": np.empty((ny, nx)), "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.002},
            "epsilon": 0.01,
            "reaction_lambda": 10.0,
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
