import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: convection_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: high
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: oscillatory_rhs
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P1
stabilization: supg
time_method: none
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: convection_diffusion
"""


def _build_problem(n, degree=1):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps = 0.05
    beta_vec = np.array([3.0, 3.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_vec.astype(np.float64))
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])

    h = ufl.CellDiameter(domain)
    beta_norm = math.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2)
    tau = h / (2.0 * beta_norm)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = f_expr * v * ufl.dx + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    return domain, V, a, L, [bc], beta_vec, eps


def _solve_once(n, degree=1, rtol=1e-9):
    domain, V, a, L, bcs, beta_vec, eps = _build_problem(n, degree)

    opts = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": rtol,
        "ksp_atol": 1e-12,
        "ksp_max_it": 4000,
    }
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options=opts, petsc_options_prefix=f"cd_{n}_"
        )
        uh = problem.solve()
        ksp = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"cdlu_{n}_"
        )
        uh = problem.solve()
        ksp = problem.solver

    uh.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    beta = fem.Constant(domain, beta_vec.astype(np.float64))
    cell_res = -eps * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - (
        ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])
    )
    residual_sq = fem.assemble_scalar(fem.form((cell_res ** 2) * ufl.dx))
    residual_sq = domain.comm.allreduce(residual_sq, op=MPI.SUM)
    residual_l2 = math.sqrt(max(residual_sq, 0.0))

    return {
        "domain": domain,
        "u": uh,
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "residual_l2": float(residual_l2),
    }


def _probe_function(u_func, pts3):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    points_on_proc = []
    cells = []
    ids = []
    for i in range(pts3.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells.append(links[0])
            ids.append(i)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(pts3.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out
    return None


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts3 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts3)
    if u_func.function_space.mesh.comm.rank == 0:
        return np.nan_to_num(vals.reshape(ny, nx), nan=0.0)
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]
    candidate_ns = [64, 96, 128]

    best = None
    prev_grid = None
    conv_indicator = None

    for i, n in enumerate(candidate_ns):
        result = _solve_once(n, degree=1, rtol=1e-9)
        u_grid = _sample_on_grid(result["u"], grid_spec)

        if comm.rank == 0:
            if prev_grid is not None:
                conv_indicator = float(np.linalg.norm(u_grid - prev_grid) / max(np.linalg.norm(u_grid), 1e-14))
            prev_grid = u_grid.copy()

        conv_indicator = comm.bcast(conv_indicator, root=0)
        best = (result, u_grid)

        elapsed = time.perf_counter() - t0
        if i > 0 and conv_indicator is not None and conv_indicator < 5e-3:
            break
        if elapsed > 30.0:
            break

    result, u_grid = best
    solver_info = {
        "mesh_resolution": result["mesh_resolution"],
        "element_degree": result["element_degree"],
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": result["rtol"],
        "iterations": result["iterations"],
        "accuracy_verification": {
            "residual_l2": result["residual_l2"],
            "grid_convergence_indicator": conv_indicator,
            "stabilization": "SUPG",
        },
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
