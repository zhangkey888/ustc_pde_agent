"""
DIAGNOSIS
- equation_type: convection_diffusion
- spatial_dim: 2
- domain_geometry: rectangle
- unknowns: scalar
- coupling: none
- linearity: linear
- time_dependence: transient
- stiffness: stiff
- dominant_physics: mixed
- peclet_or_reynolds: high
- solution_regularity: smooth
- bc_type: all_dirichlet
- special_notes: none

METHOD
- spatial_method: fem
- element_or_basis: Lagrange_P1
- stabilization: supg
- time_method: backward_euler
- nonlinear_solver: none
- linear_solver: gmres
- preconditioner: ilu
- special_treatment: none
- pde_skill: convection_diffusion
"""

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func: fem.Function, nx: int, ny: int, bbox):
    domain = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc, cells_on_proc, eval_map = [], [], []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values_local[np.array(eval_map, dtype=np.int32)] = vals
    gathered = domain.comm.allgather(values_local)
    values_global = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(values_global) & np.isfinite(arr)
        values_global[mask] = arr[mask]
    return np.nan_to_num(values_global, nan=0.0).reshape(ny, nx)


def _tau_supg(h, beta, eps_value, dt):
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-14)
    ta = h / (2.0 * beta_norm)
    td = h * h / (4.0 * eps_value + 1.0e-14)
    tt = dt
    return 1.0 / ufl.sqrt(ta**-2 + td**-2 + tt**-2 + 1.0e-14)


def _run_case(nx, degree, dt_target, t_end, out_grid):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    eps_value = 0.05
    beta_vec = np.array([2.0, 1.0], dtype=np.float64)
    n_steps = max(1, int(math.ceil(t_end / dt_target)))
    dt = float(t_end / n_steps)
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, ScalarType(beta_vec))
    f_expr = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    u_prev = fem.Function(V)
    u_prev.x.array[:] = 0.0
    u_sol = fem.Function(V)
    h = ufl.CellDiameter(domain)
    tau = _tau_supg(h, beta, eps_value, dt)
    sv = ufl.dot(beta, ufl.grad(v))
    a = ((u / dt) * v * ufl.dx + eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx + tau * ((u / dt) + ufl.dot(beta, ufl.grad(u))) * sv * ufl.dx)
    L = ((u_prev / dt) * v * ufl.dx + f_expr * v * ufl.dx + tau * ((u_prev / dt) + f_expr) * sv * ufl.dx)
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=2000)
    solver.setFromOptions()
    total_iterations = 0
    start = time.perf_counter()
    try:
        for _ in range(n_steps):
            with b.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            solver.solve(b, u_sol.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("iterative solver failed")
            u_sol.x.scatter_forward()
            total_iterations += int(solver.getIterationNumber())
            u_prev.x.array[:] = u_sol.x.array
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        total_iterations = 0
        u_prev.x.array[:] = 0.0
        for _ in range(n_steps):
            with b.localForm() as loc:
                loc.set(0.0)
            petsc.assemble_vector(b, L_form)
            petsc.apply_lifting(b, [a_form], bcs=[[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, [bc])
            solver.solve(b, u_sol.x.petsc_vec)
            u_sol.x.scatter_forward()
            u_prev.x.array[:] = u_sol.x.array
    elapsed = time.perf_counter() - start
    u_grid = _sample_function_on_grid(u_sol, out_grid["nx"], out_grid["ny"], out_grid["bbox"])
    return u_grid, {"mesh_resolution": int(nx), "element_degree": int(degree), "ksp_type": str(solver.getType()), "pc_type": str(solver.getPC().getType()), "rtol": 1e-8, "iterations": int(total_iterations), "dt": float(dt), "n_steps": int(n_steps), "time_scheme": "backward_euler", "_wall": float(elapsed)}


def solve(case_spec: dict) -> dict:
    out_grid = case_spec["output"]["grid"]
    time_spec = case_spec.get("pde", {}).get("time", {})
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.02))
    candidates = [(48, 1, min(dt_suggested, 0.01)), (72, 1, min(dt_suggested, 0.005)), (96, 1, min(dt_suggested, 0.0025)), (128, 1, min(dt_suggested, 0.0015))]
    budget = 185.909
    start_total = time.perf_counter()
    best_u = None
    best_info = None
    previous_u = None
    verification = {}
    for i, (nx, degree, dt) in enumerate(candidates):
        if time.perf_counter() - start_total > 0.9 * budget:
            break
        u_grid, info = _run_case(nx, degree, dt, t_end, out_grid)
        if previous_u is not None:
            verification[f"grid_refinement_rel_change_{i}"] = float(np.linalg.norm(u_grid - previous_u) / max(np.linalg.norm(u_grid), 1e-14))
        previous_u = u_grid.copy()
        best_u = u_grid
        best_info = info
        if time.perf_counter() - start_total > 0.5 * budget:
            break
    if best_u is None:
        raise RuntimeError("No successful candidate run")
    best_info.pop("_wall", None)
    best_info["verification"] = verification
    return {"u": np.asarray(best_u, dtype=np.float64).reshape(out_grid["ny"], out_grid["nx"]), "u_initial": np.zeros((out_grid["ny"], out_grid["nx"]), dtype=np.float64), "solver_info": best_info}


if __name__ == "__main__":
    case_spec = {"pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02}}, "output": {"grid": {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
