import time
from typing import Dict, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      transient
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   high
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P1
# stabilization:        supg
# time_method:          backward_euler
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion / reaction_diffusion / biharmonic
# ```

ScalarType = PETSc.ScalarType


def _get_case_values(case_spec: dict) -> Tuple[float, np.ndarray, float, float, float]:
    pde = case_spec.get("pde", {})
    params = case_spec.get("params", {})
    eps = params.get("epsilon", params.get("diffusion", pde.get("epsilon", pde.get("diffusion", 0.02))))
    beta = params.get("beta", params.get("velocity", pde.get("beta", pde.get("velocity", [6.0, 2.0]))))
    time_spec = pde.get("time", case_spec.get("time", {}))
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt = float(time_spec.get("dt", 0.02))
    return float(eps), np.array(beta, dtype=np.float64), t0, t_end, dt


def _uniform_grid_from_spec(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    return nx, ny, pts


def _sample_function(domain, uh: fem.Function, points_xyz: np.ndarray) -> np.ndarray:
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(domain, candidates, points_xyz)

    point_ids = []
    cells = []
    pts_local = []
    for i, pt in enumerate(points_xyz):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            cells.append(links[0])
            pts_local.append(pt)

    vals_local = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    if pts_local:
        vals = uh.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals_local[np.asarray(point_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    gathered = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        out = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        out = np.nan_to_num(out, nan=0.0)
    else:
        out = None
    return comm.bcast(out, root=0)


def _solve_once(case_spec: dict, mesh_resolution: int, dt: float, sample_points: np.ndarray) -> Dict:
    comm = MPI.COMM_WORLD
    eps, beta, t0, t_end, _ = _get_case_values(case_spec)

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    u_h = fem.Function(V)

    x = ufl.SpatialCoordinate(domain)
    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(eps))
    beta_c = fem.Constant(domain, np.asarray(beta, dtype=ScalarType))
    t_c = fem.Constant(domain, ScalarType(t0 + dt))

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-16)
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)) * ufl.exp(-t_c)

    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (12.0 * eps_c / h**2) ** 2)
    Lu = u / dt_c - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    test_supg = ufl.dot(beta_c, ufl.grad(v))

    a = (u / dt_c) * v * ufl.dx + eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx + tau * Lu * test_supg * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx + tau * ((u_n / dt_c) + f_expr) * test_supg * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1.0e-8, atol=1.0e-12, max_it=3000)
    solver.setFromOptions()

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    t = t0
    masses = []
    l2_increments = []

    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        u_old_arr = u_n.x.array.copy()
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        inc_local = np.linalg.norm(u_h.x.array - u_old_arr)
        inc = comm.allreduce(inc_local, op=MPI.SUM)
        mass_local = fem.assemble_scalar(fem.form(u_h * ufl.dx))
        mass = comm.allreduce(mass_local, op=MPI.SUM)

        masses.append(float(mass))
        l2_increments.append(float(inc))
        u_n.x.array[:] = u_h.x.array

    sampled = _sample_function(domain, u_h, sample_points)

    return {
        "u_grid_flat": sampled,
        "iterations": total_iterations,
        "mesh_resolution": int(mesh_resolution),
        "element_degree": 1,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-8,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification": {
            "mass_final": masses[-1] if masses else 0.0,
            "last_step_increment": l2_increments[-1] if l2_increments else 0.0,
            "mass_history": masses,
            "increment_history": l2_increments,
        },
    }


def solve(case_spec: dict) -> dict:
    _, _, _, _, dt_suggested = _get_case_values(case_spec)
    nx_out, ny_out, sample_points = _uniform_grid_from_spec(case_spec)

    start = time.perf_counter()
    candidates = [(64, dt_suggested), (80, dt_suggested), (96, dt_suggested / 2.0)]

    best = None
    prev_grid = None
    best_verification = {}

    for mesh_resolution, dt in candidates:
        run = _solve_once(case_spec, mesh_resolution, dt, sample_points)
        if prev_grid is not None:
            diff = np.linalg.norm(run["u_grid_flat"] - prev_grid) / (np.linalg.norm(run["u_grid_flat"]) + 1.0e-14)
            run["verification"]["relative_change_to_previous"] = float(diff)
        else:
            run["verification"]["relative_change_to_previous"] = None
        best = run
        best_verification = run["verification"]
        prev_grid = run["u_grid_flat"].copy()
        if time.perf_counter() - start > 100.0:
            break

    u_grid = best["u_grid_flat"].reshape(ny_out, nx_out)
    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.zeros((ny_out, nx_out), dtype=np.float64),
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "dt": float(best["dt"]),
            "n_steps": int(best["n_steps"]),
            "time_scheme": str(best["time_scheme"]),
            "accuracy_verification": best_verification,
        },
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02}},
        "params": {"epsilon": 0.02, "beta": [6.0, 2.0]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
