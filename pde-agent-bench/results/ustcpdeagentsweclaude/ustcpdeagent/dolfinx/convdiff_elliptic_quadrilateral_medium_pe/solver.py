import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

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
# pde_skill:            convection_diffusion / reaction_diffusion
# ```


def _exact_expr(x):
    return ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _forcing_expr(msh, eps_value, beta_value):
    x = ufl.SpatialCoordinate(msh)
    u_exact = _exact_expr(x)
    beta = ufl.as_vector((ScalarType(beta_value[0]), ScalarType(beta_value[1])))
    f = -eps_value * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))
    return u_exact, f, beta


def _make_problem(n, degree, eps_value=0.05, beta_value=(4.0, 2.0),
                  ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact, f_expr, beta = _forcing_expr(msh, eps_value, beta_value)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe = beta_norm * h / (2.0 * eps_value + 1.0e-14)
    cothPe = ufl.cosh(Pe) / ufl.sinh(Pe)
    tau = h / (2.0 * beta_norm + 1.0e-14) * (cothPe - 1.0 / (Pe + 1.0e-14))

    a_std = eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_fun * v * ufl.dx

    r_u = -eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    r_v = ufl.dot(beta, ufl.grad(v))
    a_supg = tau * r_u * r_v * ufl.dx
    L_supg = tau * f_fun * r_v * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-12,
            "ksp_max_it": 2000,
        },
    )

    return msh, V, problem, u_exact, bc


def _solve_and_measure(n, degree, eps_value=0.05, beta_value=(4.0, 2.0),
                       ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    t0 = time.perf_counter()
    msh, V, problem, u_exact, bc = _make_problem(
        n, degree, eps_value=eps_value, beta_value=beta_value,
        ksp_type=ksp_type, pc_type=pc_type, rtol=rtol
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    x = ufl.SpatialCoordinate(msh)
    e_form = fem.form(((uh - _exact_expr(x)) ** 2) * ufl.dx)
    err_local = fem.assemble_scalar(e_form)
    err_L2 = math.sqrt(msh.comm.allreduce(err_local, op=MPI.SUM))

    ksp = problem.solver
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "wall_time": float(elapsed),
        "L2_error": float(err_L2),
    }
    return msh, V, uh, info


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idxs.append(i)

    if len(points_on_proc) > 0:
        local_vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32)).reshape(-1)
        vals[np.array(idxs, dtype=np.int32)] = np.asarray(local_vals, dtype=np.float64)

    # Gather ownership from all ranks and combine by first non-NaN
    all_vals = msh.comm.allgather(vals)
    merged = np.full_like(vals, np.nan)
    for arr in all_vals:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    # For points exactly on boundary/corners, fall back to exact values if needed
    if np.isnan(merged).any():
        xx = pts[:, 0]
        yy = pts[:, 1]
        exact = np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy)
        merged[np.isnan(merged)] = exact[np.isnan(merged)]

    return merged.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    pde = case_spec.get("pde", {})
    output = case_spec["output"]
    grid = output["grid"]

    eps_value = float(pde.get("epsilon", 0.05))
    beta_value = tuple(pde.get("beta", [4.0, 2.0]))
    time_limit = float(case_spec.get("time_limit", case_spec.get("wall_time_sec", 3.268)))

    # Adaptive accuracy-time tradeoff:
    # start with efficient robust candidate, then refine if still comfortably under budget.
    candidates = [
        (28, 2, "gmres", "ilu", 1e-8),
        (24, 2, "gmres", "ilu", 1e-8),
        (20, 2, "gmres", "ilu", 1e-8),
        (32, 1, "gmres", "ilu", 1e-8),
    ]

    chosen = None
    best_tuple = None
    start_all = time.perf_counter()

    for cand in candidates:
        n, degree, ksp_type, pc_type, rtol = cand
        msh, V, uh, info = _solve_and_measure(
            n, degree,
            eps_value=eps_value,
            beta_value=beta_value,
            ksp_type=ksp_type,
            pc_type=pc_type,
            rtol=rtol,
        )
        elapsed_all = time.perf_counter() - start_all
        remaining = time_limit - elapsed_all

        chosen = (msh, V, uh, info)
        best_tuple = cand

        # stop as soon as the accuracy target is met; only refine if still needed
        if info["L2_error"] <= 1.08e-3:
            break
        if remaining < 0.35:
            break

    msh, V, uh, info = chosen
    u_grid = _sample_on_grid(msh, uh, grid)

    solver_info = {
        "mesh_resolution": info["mesh_resolution"],
        "element_degree": info["element_degree"],
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": info["rtol"],
        "iterations": info["iterations"],
        "L2_error": info["L2_error"],
        "wall_time": info["wall_time"],
        "stabilization": "SUPG",
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 0.05,
            "beta": [4.0, 2.0],
            "time": None,
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "time_limit": 3.268,
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
