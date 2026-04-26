import math
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   high
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        supg
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion / reaction_diffusion / biharmonic
# ```

ScalarType = PETSc.ScalarType


def _u_exact_expr(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _beta_vector():
    return np.array([10.0, 4.0], dtype=np.float64)


def _manufactured_rhs(x, eps):
    uex = _u_exact_expr(x)
    lap_u = -(ufl.pi**2 + (2.0 * ufl.pi) ** 2) * uex
    grad_u = ufl.as_vector(
        [
            ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1]),
            2.0 * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1]),
        ]
    )
    beta = ufl.as_vector((ScalarType(10.0), ScalarType(4.0)))
    return -eps * lap_u + ufl.dot(beta, grad_u)


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([xx.ravel(), yy.ravel()])
    pts3 = np.column_stack([pts2, np.zeros((pts2.shape[0], 1), dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    values_local = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        values_local[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    comm = domain.comm
    values_global = np.empty_like(values_local)
    comm.Allreduce(values_local, values_global, op=MPI.MAX)

    return values_global.reshape(ny, nx)


def _solve_once(mesh_resolution: int, degree: int, ksp_type="gmres", pc_type="ilu", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps_val = 0.01
    beta_np = _beta_vector()
    beta = ufl.as_vector((ScalarType(beta_np[0]), ScalarType(beta_np[1])))
    beta_norm = float(np.linalg.norm(beta_np))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact_ufl = _u_exact_expr(x)
    f_ufl = _manufactured_rhs(x, eps_val)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    h = ufl.CellDiameter(domain)
    beta_dot_grad_u = ufl.dot(beta, ufl.grad(u))
    beta_dot_grad_v = ufl.dot(beta, ufl.grad(v))
    residual_strong = -eps_val * ufl.div(ufl.grad(u)) + beta_dot_grad_u

    # SUPG parameter
    tau = h / (2.0 * beta_norm)
    PeK = beta_norm * h / (2.0 * eps_val)
    tau = ufl.conditional(ufl.gt(PeK, 1.0), tau, h * h / (4.0 * eps_val + 1.0e-14))

    a = (
        eps_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * residual_strong * beta_dot_grad_v * ufl.dx
    )
    L = (
        f_ufl * v * ufl.dx
        + tau * f_ufl * beta_dot_grad_v * ufl.dx
    )

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 20000,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    solve_time = time.perf_counter() - t0
    uh.x.scatter_forward()

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    actual_ksp = ksp.getType()
    actual_pc = ksp.getPC().getType()

    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_err = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx)), op=MPI.SUM))
    h1_semi = math.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(err_fun), ufl.grad(err_fun)) * ufl.dx)), op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "l2_error": l2_err,
        "h1_semi_error": h1_semi,
        "solve_time": solve_time,
        "iterations": its,
        "ksp_type": actual_ksp,
        "pc_type": actual_pc,
        "rtol": rtol,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
    }


def solve(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    comm = MPI.COMM_WORLD

    candidates = [
        (96, 2),
        (128, 2),
        (160, 2),
        (192, 2),
    ]

    best = None
    wall_budget = 41.866
    start = time.perf_counter()

    for n, p in candidates:
        result = _solve_once(n, p, ksp_type="gmres", pc_type="ilu", rtol=1.0e-10)
        best = result
        elapsed = time.perf_counter() - start
        remaining = wall_budget - elapsed
        if remaining < 8.0:
            break
        if result["l2_error"] <= 2.48e-05 and elapsed > 0.55 * wall_budget:
            break

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["domain"], best["uh"], grid)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "verification": {
            "manufactured_solution": "sin(pi*x)*sin(2*pi*y)",
            "l2_error": float(best["l2_error"]),
            "h1_semi_error": float(best["h1_semi_error"]),
            "solve_time_sec": float(best["solve_time"]),
            "supg": True,
            "epsilon": 0.01,
            "beta": [10.0, 4.0],
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
