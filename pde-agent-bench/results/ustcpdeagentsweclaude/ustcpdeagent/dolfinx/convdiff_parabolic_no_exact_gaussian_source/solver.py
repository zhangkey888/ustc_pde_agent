import math
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
# special_notes:        variable_coeff
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


def _time_spec(case_spec: dict) -> Tuple[float, float, float]:
    pde = case_spec.get("pde", {})
    tinfo = pde.get("time", case_spec.get("time", {}))
    t0 = float(tinfo.get("t0", 0.0))
    t_end = float(tinfo.get("t_end", 0.1))
    dt = float(tinfo.get("dt", 0.02))
    return t0, t_end, dt


def _params(case_spec: dict) -> Tuple[float, np.ndarray]:
    pde = case_spec.get("pde", {})
    params = case_spec.get("params", {})
    eps = params.get("epsilon", params.get("diffusion", pde.get("epsilon", pde.get("diffusion", 0.02))))
    beta = params.get("beta", params.get("velocity", pde.get("beta", pde.get("velocity", [6.0, 2.0]))))
    return float(eps), np.asarray(beta, dtype=np.float64)


def _output_grid(case_spec: dict) -> Tuple[int, int, np.ndarray]:
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

    point_ids, cells, pts = [], [], []
    for i, pt in enumerate(points_xyz):
        links = colliding.links(i)
        if len(links) > 0:
            point_ids.append(i)
            cells.append(links[0])
            pts.append(pt)

    vals_local = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
    if pts:
        vals = uh.eval(np.asarray(pts, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals_local[np.asarray(point_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    gathered = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        vals = np.full(points_xyz.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        vals = np.nan_to_num(vals, nan=0.0)
    else:
        vals = None
    return comm.bcast(vals, root=0)


def _make_zero_bc(domain, V):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _build_ksp(comm, A, ksp_type="gmres", pc_type="ilu", rtol=1.0e-8):
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()
    return solver


def _run(case_spec: dict, mesh_resolution: int, dt_in: float, points: np.ndarray, degree: int = 1) -> Dict:
    comm = MPI.COMM_WORLD
    eps, beta = _params(case_spec)
    t0, t_end, _ = _time_spec(case_spec)

    n_steps = max(1, int(math.ceil((t_end - t0) / dt_in)))
    dt = (t_end - t0) / n_steps

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    bc = _make_zero_bc(domain, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    uh = fem.Function(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0

    dt_c = fem.Constant(domain, ScalarType(dt))
    eps_c = fem.Constant(domain, ScalarType(eps))
    beta_c = fem.Constant(domain, np.asarray(beta, dtype=ScalarType))
    t_c = fem.Constant(domain, ScalarType(t0))
    x = ufl.SpatialCoordinate(domain)

    source = ufl.exp(-200.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2)) * ufl.exp(-t_c)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1.0e-16)
    tau = 1.0 / ufl.sqrt((2.0 / dt_c) ** 2 + (2.0 * beta_norm / h) ** 2 + (12.0 * eps_c / (h * h)) ** 2)

    test_supg = ufl.dot(beta_c, ufl.grad(v))

    a = (
        (u / dt_c) * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
        + tau * ((u / dt_c) + ufl.dot(beta_c, ufl.grad(u))) * test_supg * ufl.dx
    )
    L = (
        (u_n / dt_c) * v * ufl.dx
        + source * v * ufl.dx
        + tau * ((u_n / dt_c) + source) * test_supg * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    solver = _build_ksp(domain.comm, A, ksp_type="gmres", pc_type="ilu", rtol=1.0e-8)
    b = petsc.create_vector(L_form.function_spaces)

    total_iterations = 0
    masses = []
    increments = []

    t = t0
    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        old_arr = u_n.x.array.copy()
        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.destroy()
            solver = _build_ksp(domain.comm, A, ksp_type="preonly", pc_type="lu", rtol=1.0e-12)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        diff_local = np.dot(uh.x.array - old_arr, uh.x.array - old_arr)
        inc = math.sqrt(comm.allreduce(diff_local, op=MPI.SUM))
        mass_local = fem.assemble_scalar(fem.form(uh * ufl.dx))
        mass = comm.allreduce(mass_local, op=MPI.SUM)

        increments.append(float(inc))
        masses.append(float(mass))
        u_n.x.array[:] = uh.x.array

    sampled = _sample_function(domain, uh, points)
    u_l2_local = fem.assemble_scalar(fem.form(uh * uh * ufl.dx))
    grad_l2_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx))
    u_l2 = math.sqrt(comm.allreduce(u_l2_local, op=MPI.SUM))
    grad_l2 = math.sqrt(comm.allreduce(grad_l2_local, op=MPI.SUM))

    return {
        "u_flat": sampled,
        "iterations": int(total_iterations),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1.0e-8,
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification": {
            "mass_history": masses,
            "increment_history": increments,
            "mass_final": masses[-1] if masses else 0.0,
            "last_step_increment": increments[-1] if increments else 0.0,
            "solution_l2_norm": float(u_l2),
            "solution_h1_semi_norm": float(grad_l2),
        },
    }


def solve(case_spec: dict) -> dict:
    nx, ny, points = _output_grid(case_spec)
    _, _, dt_suggest = _time_spec(case_spec)

    budget = 330.0
    start = time.perf_counter()

    candidates = [
        (72, min(dt_suggest, 0.01)),
        (96, 0.01),
        (128, 0.005),
        (160, 0.005),
    ]

    best = None
    previous = None
    verification = {}

    for mesh_resolution, dt in candidates:
        elapsed = time.perf_counter() - start
        if elapsed > budget:
            break

        run = _run(case_spec, mesh_resolution, dt, points, degree=1)
        if previous is not None:
            diff = np.linalg.norm(run["u_flat"] - previous) / (np.linalg.norm(run["u_flat"]) + 1.0e-14)
            run["verification"]["relative_change_to_previous"] = float(diff)
        else:
            run["verification"]["relative_change_to_previous"] = None

        best = run
        verification = run["verification"]
        previous = run["u_flat"].copy()

        elapsed = time.perf_counter() - start
        if elapsed > 0.85 * budget:
            break

    u_grid = best["u_flat"].reshape(ny, nx)
    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "u_initial": np.zeros((ny, nx), dtype=np.float64),
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
            "accuracy_verification": verification,
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
