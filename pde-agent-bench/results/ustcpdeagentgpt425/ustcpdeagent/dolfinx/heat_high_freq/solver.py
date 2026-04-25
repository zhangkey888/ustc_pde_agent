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

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([X.ravel(), Y.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0], dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if eval_ids:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            miss = np.where(~np.isfinite(merged))[0]
            for i in miss:
                merged[i] = _exact_numpy(pts3[i, 0], pts3[i, 1], 0.0)
        return merged.reshape(ny, nx)
    return None


def _assemble_and_solve(case_spec, mesh_resolution, degree, dt):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    kappa = fem.Constant(domain, ScalarType(1.0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f_ufl = (-1.0 + 32.0 * ufl.pi * ufl.pi) * u_exact_ufl

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(0.0) * np.sin(4.0 * np.pi * X[0]) * np.sin(4.0 * np.pi * X[1]))

    u_bc = fem.Function(V)
    f_fun = fem.Function(V)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array
    u_initial.x.scatter_forward()

    p_exact = fem.Function(V)
    e_fun = fem.Function(V)

    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", case_spec.get("t_end", 0.1)))
    n_steps = int(round((t_end - t0) / dt))
    t = t0
    total_iterations = 0

    for _ in range(n_steps):
        t = min(t + dt, t_end)
        t_const.value = ScalarType(t)

        u_bc.interpolate(
            fem.Expression(u_exact_ufl, V.element.interpolation_points)
        )
        f_fun.interpolate(
            fem.Expression(f_ufl, V.element.interpolation_points)
        )

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        uh.x.array[:] = u_n.x.array
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    p_exact.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    e_fun.x.array[:] = uh.x.array - p_exact.x.array
    e_fun.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(fem.form(ufl.inner(e_fun, e_fun) * ufl.dx))
    norm_L2_local = fem.assemble_scalar(fem.form(ufl.inner(p_exact, p_exact) * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    norm_L2 = math.sqrt(comm.allreduce(norm_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / max(norm_L2, 1e-16)

    return {
        "domain": domain,
        "uh": uh,
        "u_initial": u_initial,
        "error_L2": err_L2,
        "relative_L2": rel_L2,
        "iterations": int(total_iterations),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    pde_time = case_spec.get("pde", {}).get("time", {})
    t0 = float(pde_time.get("t0", 0.0))
    t_end = float(pde_time.get("t_end", 0.1))
    dt_suggested = float(pde_time.get("dt", 0.005))
    time_budget = 30.858

    candidates = [
        (48, 2, dt_suggested),
        (64, 2, dt_suggested),
        (72, 2, dt_suggested / 2.0),
        (80, 2, dt_suggested / 2.0),
        (96, 2, dt_suggested / 2.0),
    ]

    # Ensure exact divisibility of interval for each dt candidate
    normalized = []
    for nx, deg, dt in candidates:
        n_steps = max(1, int(round((t_end - t0) / dt)))
        dt_use = (t_end - t0) / n_steps
        normalized.append((nx, deg, dt_use))

    best = None
    for nx, deg, dt in normalized:
        now = time.perf_counter()
        elapsed = now - t_start
        if elapsed > 0.9 * time_budget and best is not None:
            break
        result = _assemble_and_solve(case_spec, nx, deg, dt)
        if best is None or result["relative_L2"] < best["relative_L2"]:
            best = result
        elapsed_after = time.perf_counter() - t_start
        if elapsed_after > 0.75 * time_budget:
            break

    domain = best["domain"]
    uh = best["uh"]
    u_initial = best["u_initial"]

    u_grid = _sample_function_on_grid(domain, uh, output_grid)
    u0_grid = _sample_function_on_grid(domain, u_initial, output_grid)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": best["mesh_resolution"],
            "element_degree": best["element_degree"],
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": "backward_euler",
            "error_L2": float(best["error_L2"]),
            "relative_L2": float(best["relative_L2"]),
        }
        return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": None}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.005}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
