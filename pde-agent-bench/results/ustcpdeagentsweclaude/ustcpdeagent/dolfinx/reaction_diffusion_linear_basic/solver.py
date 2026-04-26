import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: reaction_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: reaction_diffusion
"""


def _get_time_params(case_spec: Dict[str, Any]) -> Tuple[float, float, float, str]:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.5))
    dt = float(time_spec.get("dt", 0.01))
    scheme = str(time_spec.get("scheme", "backward_euler"))
    return t0, t_end, dt, scheme


def _extract_epsilon(case_spec: Dict[str, Any]) -> float:
    pde = case_spec.get("pde", {})
    for key in ("epsilon", "eps", "diffusivity", "nu", "kappa"):
        if key in pde:
            return float(pde[key])
    return 0.1


def _uniform_grid(case_spec: Dict[str, Any]) -> Tuple[int, int, Tuple[float, float, float, float]]:
    out = case_spec.get("output", {}).get("grid", {})
    nx = int(out.get("nx", 64))
    ny = int(out.get("ny", 64))
    bbox = out.get("bbox", [0.0, 1.0, 0.0, 1.0])
    return nx, ny, (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _manufactured_exact_expr(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _source_expr(msh, epsilon, t):
    x = ufl.SpatialCoordinate(msh)
    uex = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    return (2.0 * epsilon * ufl.pi**2) * uex


def _probe_points(u_func: fem.Function, points_array: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    pts = points_array.T
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int64)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_grid(u_func: fem.Function, nx: int, ny: int, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    local_vals = _probe_points(u_func, pts)

    comm = u_func.function_space.mesh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged, nan=0.0)
        grid = merged.reshape(ny, nx)
    else:
        grid = None

    return comm.bcast(grid, root=0)


def _run_solver_once(case_spec: Dict[str, Any], mesh_resolution: int, degree: int, dt: float,
                     ksp_type: str, pc_type: str, rtol: float) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    t0, t_end, _, scheme = _get_time_params(case_spec)
    if scheme.lower() != "backward_euler":
        scheme = "backward_euler"
    epsilon = _extract_epsilon(case_spec)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(_manufactured_exact_expr(msh, ScalarType(t0)), V.element.interpolation_points))

    u_bc = fem.Function(V)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_c = fem.Constant(msh, ScalarType(dt))
    eps_c = fem.Constant(msh, ScalarType(epsilon))
    t_c = fem.Constant(msh, ScalarType(t0 + dt))

    f_expr = _source_expr(msh, eps_c, t_c)

    a = (u * v + dt_c * eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + dt_c * u * v) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=2000)

    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        tnow = t0 + step * dt
        t_c.value = ScalarType(tnow)
        u_bc.interpolate(fem.Expression(_manufactured_exact_expr(msh, ScalarType(tnow)), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
            if solver.getConvergedReason() <= 0:
                raise RuntimeError("Iterative solve failed")
            total_iterations += max(0, int(solver.getIterationNumber()))
        except Exception:
            lu = PETSc.KSP().create(comm)
            lu.setOperators(A)
            lu.setType("preonly")
            lu.getPC().setType("lu")
            lu.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start

    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(_manufactured_exact_expr(msh, ScalarType(t_end)), V.element.interpolation_points))
    err_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    norm_local = fem.assemble_scalar(fem.form((u_ex) ** 2 * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    norm_l2 = math.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
    rel_l2 = err_l2 / max(norm_l2, 1e-16)

    nx, ny, bbox = _uniform_grid(case_spec)
    u_grid = _sample_on_grid(uh, nx, ny, bbox)

    u0_fun = fem.Function(V)
    u0_fun.interpolate(fem.Expression(_manufactured_exact_expr(msh, ScalarType(t0)), V.element.interpolation_points))
    u0_grid = _sample_on_grid(u0_fun, nx, ny, bbox)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "accuracy_verification": {
                "manufactured_solution": "exp(-t)*sin(pi*x)*sin(pi*y)",
                "l2_error": float(err_l2),
                "relative_l2_error": float(rel_l2),
                "wall_time_sec_solver_loop": float(wall),
            },
        },
    }


def solve(case_spec: dict) -> dict:
    _, _, dt_suggested, _ = _get_time_params(case_spec)
    time_budget = float(case_spec.get("time_limit", case_spec.get("wall_time_limit", 94.701)) or 94.701)

    candidates = [
        {"mesh_resolution": 48, "degree": 1, "dt": min(dt_suggested, 0.01), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 64, "degree": 1, "dt": min(dt_suggested, 0.01), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 64, "degree": 2, "dt": min(dt_suggested, 0.01), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
        {"mesh_resolution": 80, "degree": 2, "dt": min(dt_suggested, 0.005), "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10},
    ]

    best = None
    start = time.perf_counter()

    for cfg in candidates:
        if (time.perf_counter() - start) > 0.85 * time_budget and best is not None:
            break
        try:
            res = _run_solver_once(case_spec, **cfg)
            best = res
        except Exception:
            continue

    if best is None:
        best = _run_solver_once(
            case_spec,
            mesh_resolution=40,
            degree=1,
            dt=min(dt_suggested, 0.01),
            ksp_type="preonly",
            pc_type="lu",
            rtol=1e-10,
        )

    return best
