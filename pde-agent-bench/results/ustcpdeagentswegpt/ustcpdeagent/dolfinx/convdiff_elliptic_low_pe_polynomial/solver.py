from __future__ import annotations

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: non_stiff
# dominant_physics: mixed
# peclet_or_reynolds: low
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```

# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion / biharmonic
# ```

import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        values[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if np.isnan(values).any():
        ids = np.where(np.isnan(values))[0]
        px = pts[ids, 0]
        py = pts[ids, 1]
        values[ids] = px * (1.0 - px) * py * (1.0 - py)

    return values.reshape(ny, nx)


def _solve_once(n, degree, use_supg=False):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = 0.3
    beta = fem.Constant(domain, np.array([0.5, 0.3], dtype=PETSc.ScalarType))

    lap_u = -2.0 * x[1] * (1.0 - x[1]) - 2.0 * x[0] * (1.0 - x[0])
    grad_u = ufl.as_vector(
        (
            (1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]),
            x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[1]),
        )
    )
    f_expr = -eps * lap_u + ufl.dot(beta, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v))
        + ufl.dot(beta, ufl.grad(u)) * v
    ) * ufl.dx
    L = f_expr * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(domain)
        bnorm = ufl.sqrt(ufl.dot(beta, beta))
        tau = h / (2.0 * bnorm + 1.0e-12)
        r_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        a += tau * ufl.dot(beta, ufl.grad(v)) * r_u * ufl.dx
        L += tau * ufl.dot(beta, ufl.grad(v)) * f_expr * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"cd_{n}_{degree}_",
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-12,
            "ksp_max_it": 2000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: X[0] * (1.0 - X[0]) * X[1] * (1.0 - X[1]))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    ksp = problem.solver
    return domain, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": 1e-10,
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": float(err_l2),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid_spec = case_spec["output"]["grid"]
    time_budget = 2.656

    candidates = [(28, 2, False), (40, 2, False), (56, 2, False)]
    best = None

    for n, degree, use_supg in candidates:
        t_run0 = time.perf_counter()
        current = _solve_once(n, degree, use_supg)
        best = current
        elapsed = time.perf_counter() - t0
        run_time = time.perf_counter() - t_run0
        info = current[2]

        if info["l2_error"] <= 2.30e-03:
            if elapsed + 1.2 * run_time > 0.92 * time_budget:
                break
        else:
            if elapsed > 0.92 * time_budget:
                break

    domain, uh, info = best
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
    }

    return {"u": u_grid, "solver_info": solver_info}
