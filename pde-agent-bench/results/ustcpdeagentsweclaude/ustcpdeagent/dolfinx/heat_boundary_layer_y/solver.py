import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

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
# solution_regularity: boundary_layer
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
COMM = MPI.COMM_WORLD


def _infer_time_data(case_spec: Dict[str, Any]):
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.08)))
    dt_suggested = float(pde.get("dt", case_spec.get("dt", 0.008)))
    scheme = pde.get("scheme", case_spec.get("scheme", "backward_euler"))
    return t0, t_end, dt_suggested, scheme


def _manufactured_u_expr(x, t):
    return ufl.exp(-t) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])


def _probe_function(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64),
                           np.array(local_cells, dtype=np.int32))
        local_vals[np.array(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        result = np.full(points.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            result[mask] = arr[mask]
        return result
    return np.empty(points.shape[0], dtype=np.float64)


def _sample_on_grid(u_func: fem.Function, grid: Dict[str, Any]) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    if COMM.rank == 0:
        return vals.reshape(ny, nx)
    return np.empty((ny, nx), dtype=np.float64)


def _choose_resolution_and_dt(time_budget: float, t0: float, t_end: float, dt_suggested: float):
    if time_budget >= 3.5:
        n = 56
        degree = 2
        dt = min(dt_suggested, 0.004)
    else:
        n = 40
        degree = 2
        dt = min(dt_suggested, 0.006)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps
    return n, degree, dt, n_steps


def solve(case_spec: dict) -> dict:
    t_wall_start = time.perf_counter()

    t0, t_end, dt_suggested, scheme = _infer_time_data(case_spec)
    scheme = "backward_euler"
    budget = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 4.302)))
    mesh_resolution, degree, dt, n_steps = _choose_resolution_and_dt(budget, t0, t_end, dt_suggested)

    msh = mesh.create_unit_square(COMM, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, ScalarType(t0))
    kappa = fem.Constant(msh, ScalarType(1.0))

    u_exact_ufl = _manufactured_u_expr(x, t_const)
    f_ufl = ufl.diff(u_exact_ufl, t_const) - ufl.div(kappa * ufl.grad(u_exact_ufl))

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_manufactured_u_expr(x, ScalarType(t0)), V.element.interpolation_points))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(_manufactured_u_expr(x, t_const), V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(COMM)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0

    grid = case_spec["output"]["grid"]
    u_initial = _sample_on_grid(u_n, grid) if COMM.rank == 0 else None

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_const.value = ScalarType(t_now)
        u_bc.interpolate(fem.Expression(_manufactured_u_expr(x, t_const), V.element.interpolation_points))
        f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except RuntimeError:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 0))
        u_n.x.array[:] = uh.x.array

    u_ex_T = fem.Function(V)
    u_ex_T.interpolate(fem.Expression(_manufactured_u_expr(x, ScalarType(t_end)), V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_ex_T.x.array
    local_l2_sq = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = math.sqrt(COMM.allreduce(local_l2_sq, op=MPI.SUM))
    local_max = np.max(np.abs(err_fun.x.array)) if err_fun.x.array.size else 0.0
    linf_error = COMM.allreduce(local_max, op=MPI.MAX)

    u_grid = _sample_on_grid(uh, grid) if COMM.rank == 0 else None

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(1e-10),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification": {
            "l2_error_final": float(l2_error),
            "linf_error_nodal_final": float(linf_error),
            "wall_time_sec": float(time.perf_counter() - t_wall_start),
        },
    }

    if COMM.rank == 0:
        return {"u": u_grid, "solver_info": solver_info, "u_initial": u_initial}
    return {"u": None, "solver_info": solver_info, "u_initial": None}


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.08, "dt": 0.008, "scheme": "backward_euler", "time": True},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "time_limit": 4.302,
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
