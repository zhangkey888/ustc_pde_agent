import math
import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: convection_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: high
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: supg
time_method: backward_euler
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: convection_diffusion
"""


def _extract_time_params(case_spec: Dict[str, Any]) -> Tuple[float, float, float]:
    pde = case_spec.get("pde", {})
    t0 = float(pde.get("t0", case_spec.get("t0", 0.0)))
    t_end = float(pde.get("t_end", case_spec.get("t_end", 0.06)))
    dt = float(pde.get("dt", case_spec.get("dt", 0.005)))
    if dt <= 0:
        dt = 0.005
    return t0, t_end, dt


def _exact_numpy(x, y, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x) * np.sin(np.pi * y)


def _exact_ufl(x, t):
    return ufl.exp(-t) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _beta(domain):
    return fem.Constant(domain, np.array([12.0, 4.0], dtype=np.float64))


def _eps(domain):
    return fem.Constant(domain, ScalarType(0.01))


def _forcing_ufl(domain, t_const):
    x = ufl.SpatialCoordinate(domain)
    uex = _exact_ufl(x, t_const)
    ut = -uex
    lap = -((4.0 * ufl.pi) ** 2 + (ufl.pi) ** 2) * uex
    beta = _beta(domain)
    eps = _eps(domain)
    conv = ufl.dot(beta, ufl.grad(uex))
    return ut - eps * lap + conv


def _interp_exact(fun: fem.Function, t: float):
    fun.interpolate(lambda x: _exact_numpy(x[0], x[1], t))


def _probe(u_fun: fem.Function, points: np.ndarray) -> np.ndarray:
    domain = u_fun.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points.T)
    colliding = geometry.compute_colliding_cells(domain, candidates, points.T)

    values = np.full(points.shape[1], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids = []
    for i in range(points.shape[1]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points.T[i])
            cells_local.append(links[0])
            ids.append(i)

    if pts_local:
        vals = u_fun.eval(np.array(pts_local, dtype=np.float64),
                          np.array(cells_local, dtype=np.int32))
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_grid(u_fun: fem.Function, grid_spec: Dict[str, Any]) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe(u_fun, pts)
    return vals.reshape(ny, nx)


def _compute_errors(domain, uh: fem.Function, t: float) -> Dict[str, float]:
    V = uh.function_space
    uex = fem.Function(V)
    _interp_exact(uex, t)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - uex.x.array
    e.x.scatter_forward()

    err2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ex2_local = fem.assemble_scalar(fem.form(ufl.inner(uex, uex) * ufl.dx))
    err2 = domain.comm.allreduce(err2_local, op=MPI.SUM)
    ex2 = domain.comm.allreduce(ex2_local, op=MPI.SUM)
    l2 = math.sqrt(max(err2, 0.0))
    rel = l2 / max(math.sqrt(max(ex2, 0.0)), 1e-14)
    return {"l2_error": float(l2), "relative_l2_error": float(rel)}


def _run_simulation(nx: int, degree: int, dt: float, t0: float, t_end: float):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    beta = _beta(domain)
    eps = _eps(domain)

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_n = fem.Function(V)
    uh = fem.Function(V)
    bc_fun = fem.Function(V)
    _interp_exact(u_n, t0)

    n_steps = int(round((t_end - t0) / dt))
    t_final = t0 + n_steps * dt

    t_const = fem.Constant(domain, ScalarType(t0))
    f = _forcing_ufl(domain, t_const)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    tau = h / (2.0 * beta_norm + 4.0 * eps / h + 1e-12)

    a = (
        (u / ScalarType(dt)) * v * ufl.dx
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * (u / ScalarType(dt) + ufl.dot(beta, ufl.grad(u)) - eps * ufl.div(ufl.grad(u)))
        * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        (u_n / ScalarType(dt)) * v * ufl.dx
        + f * v * ufl.dx
        + tau * (u_n / ScalarType(dt) + f) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    total_iterations = 0

    for step in range(1, n_steps + 1):
        t_now = t0 + step * dt
        t_const.value = ScalarType(t_now)
        _interp_exact(bc_fun, t_now)
        bc = fem.dirichletbc(bc_fun, dofs)

        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()

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

    errors = _compute_errors(domain, uh, t_final)
    info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1e-9,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }
    return domain, uh, info, errors


def solve(case_spec: dict) -> dict:
    t_wall0 = time.perf_counter()
    t0, t_end, dt_req = _extract_time_params(case_spec)

    candidates = [
        {"nx": 56, "degree": 1, "dt": min(dt_req, 0.005)},
        {"nx": 72, "degree": 1, "dt": min(dt_req, 0.004)},
        {"nx": 88, "degree": 1, "dt": min(dt_req, 0.003)},
    ]

    selected = None
    best = None
    budget = 27.471
    soft_limit = 23.5

    for cfg in candidates:
        sim_t0 = time.perf_counter()
        data = _run_simulation(cfg["nx"], cfg["degree"], cfg["dt"], t0, t_end)
        sim_elapsed = time.perf_counter() - sim_t0
        domain, uh, info, errors = data
        pack = (domain, uh, info, errors, sim_elapsed)
        best = pack
        total_elapsed = time.perf_counter() - t_wall0

        if errors["relative_l2_error"] <= 4.86e-3:
            selected = pack
            if total_elapsed + sim_elapsed > soft_limit:
                break
        if total_elapsed > budget:
            break

    if selected is None:
        selected = best

    domain, uh, info, errors, _ = selected
    grid = case_spec["output"]["grid"]
    u_grid = _sample_grid(uh, grid)

    V = uh.function_space
    u0 = fem.Function(V)
    _interp_exact(u0, t0)
    u0_grid = _sample_grid(u0, grid)

    solver_info = dict(info)
    solver_info["verification_l2_error"] = float(errors["l2_error"])
    solver_info["verification_relative_l2_error"] = float(errors["relative_l2_error"])
    solver_info["wall_time_sec"] = float(time.perf_counter() - t_wall0)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"t0": 0.0, "t_end": 0.06, "dt": 0.005},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
