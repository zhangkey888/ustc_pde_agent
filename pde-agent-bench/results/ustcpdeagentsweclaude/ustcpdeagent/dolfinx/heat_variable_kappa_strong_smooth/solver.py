import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

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

def _exact_np(t: float, xarr):
    return np.exp(-t) * np.sin(3.0 * np.pi * xarr[0]) * np.sin(2.0 * np.pi * xarr[1])

def _probe_points(u_func: fem.Function, pts3: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts3.T)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts3.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts3.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((pts3.shape[1],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        merged = np.nan_to_num(merged)
    else:
        merged = None
    merged = comm.bcast(merged, root=0)
    return merged

def _compute_l2_error(u_h: fem.Function, t: float) -> float:
    V = u_h.function_space
    msh = V.mesh
    u_ex = fem.Function(V)
    u_ex.interpolate(lambda x: _exact_np(t, x))
    e = fem.Function(V)
    e.x.array[:] = u_h.x.array - u_ex.x.array
    e.x.scatter_forward()
    val_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    return float(np.sqrt(msh.comm.allreduce(val_local, op=MPI.SUM)))

def _run(case_spec: Dict[str, Any], mesh_resolution: int, element_degree: int, dt: float) -> dict:
    comm = MPI.COMM_WORLD

    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.8 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    def u_exact_ufl(tval):
        return ufl.exp(-tval) * ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    def f_exact_ufl(tval):
        uex = u_exact_ufl(tval)
        return -uex - ufl.div(kappa * ufl.grad(uex))

    u_n = fem.Function(V)
    u_n.interpolate(lambda xx: _exact_np(t0, xx))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_fun = fem.Function(V)
    u_bc = fem.Function(V)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    t_first = t0 + dt
    u_bc.interpolate(lambda xx: _exact_np(t_first, xx))
    bc = fem.dirichletbc(u_bc, dofs)

    f_expr = fem.Expression(f_exact_ufl(t_first), V.element.interpolation_points)
    f_fun.interpolate(f_expr)

    a = (u * v + dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)

    uh = fem.Function(V)
    iterations = 0

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    u_initial = _probe_points(u_n, pts3).reshape(ny, nx)

    start = time.perf_counter()
    for n in range(n_steps):
        tn1 = t0 + (n + 1) * dt
        u_bc.interpolate(lambda xx, tt=tn1: _exact_np(tt, xx))
        f_expr = fem.Expression(f_exact_ufl(tn1), V.element.interpolation_points)
        f_fun.interpolate(f_expr)

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
        if solver.getType().lower() != "preonly":
            iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    wall = time.perf_counter() - start
    l2_error = _compute_l2_error(uh, t_end)
    u_grid = _probe_points(uh, pts3).reshape(ny, nx)

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.asarray(u_initial, dtype=np.float64),
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(solver.getType()),
            "pc_type": str(solver.getPC().getType()),
            "rtol": float(1e-10),
            "iterations": int(iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
            "l2_error": float(l2_error),
            "wall_time": float(wall),
        },
    }

def solve(case_spec: dict) -> dict:
    budget = 9.703
    suggested_dt = float(case_spec.get("pde", {}).get("time", {}).get("dt", 0.01))
    candidates = [
        (40, 1, min(suggested_dt, 0.01)),
        (56, 2, min(suggested_dt, 0.01)),
        (72, 2, 0.005),
        (88, 2, 0.004),
        (104, 2, 0.0033333333333333335),
    ]

    best = None
    spent = 0.0
    for mesh_resolution, element_degree, dt in candidates:
        tic = time.perf_counter()
        result = _run(case_spec, mesh_resolution, element_degree, dt)
        spent += time.perf_counter() - tic
        if best is None or result["solver_info"]["l2_error"] < best["solver_info"]["l2_error"]:
            best = result
        if spent > 0.8 * budget:
            break

    return best

if __name__ == "__main__":
    spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01}},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
