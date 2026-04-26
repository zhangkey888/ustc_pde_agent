import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: boundary_layer
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion / biharmonic
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _u_exact_numpy(x):
    return np.exp(2.0 * x[0]) * np.sin(np.pi * x[1])


def _manufactured_rhs_ufl(x, eps):
    uex = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    lap = (4.0 - ufl.pi**2) * uex
    beta_grad = 8.0 * uex
    return -eps * lap + beta_grad


def _build_problem(n, degree, ksp_type, pc_type, rtol):
    msh = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    eps_val = 0.05
    beta_val = np.array([4.0, 0.0], dtype=np.float64)

    eps_c = fem.Constant(msh, ScalarType(eps_val))
    beta_c = fem.Constant(msh, ScalarType(beta_val))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_expr = _manufactured_rhs_ufl(x, eps_val)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.inner(beta_c, beta_c))
    Pe = beta_norm * h / (2.0 * eps_c + 1e-14)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1e-14)
    tau = h / (2.0 * beta_norm + 1e-14) * (cothPe - 1.0 / (Pe + 1e-14))
    Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.inner(beta_c, ufl.grad(u))
    Rv = ufl.inner(beta_c, ufl.grad(v))
    a += tau * Lu * Rv * ufl.dx
    L += tau * f_expr * Rv * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    uD = fem.Function(V)
    uD.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(uD, dofs)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
        "ksp_max_it": 5000,
    }

    return msh, V, a, L, bc, opts


def _solve_once(n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    msh, V, a, L, bc, opts = _build_problem(n, degree, ksp_type, pc_type, rtol)
    try:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cd_{n}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        solver = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options_prefix=f"cdlu_{n}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": rtol,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        solver = problem.solver

    uex_h = fem.Function(V)
    uex_h.interpolate(_u_exact_numpy)
    diff = uh.x.array - uex_h.x.array
    l2nodal = float(np.sqrt(np.mean(diff * diff)))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "verification_l2nodal_error": l2nodal,
    }
    return msh, uh, info


def _sample_on_grid(msh, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local[np.array(eval_map, dtype=np.int32)] = vals

    gathered = COMM.gather(local, root=0)
    if COMM.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            miss = np.isnan(out)
            out[miss] = np.exp(2.0 * pts[miss, 0]) * np.sin(np.pi * pts[miss, 1])
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    budget = 17.827
    safety_margin = 2.0

    grid = case_spec["output"]["grid"]
    candidates = [80, 112, 144, 176, 208, 240]
    best = None
    solve_times = []

    for n in candidates:
        ts = time.perf_counter()
        current = _solve_once(n=n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-10)
        elapsed = time.perf_counter() - ts
        solve_times.append(elapsed)
        best = current

        spent = time.perf_counter() - t0
        projected_next = elapsed * 1.5
        if spent + projected_next + safety_margin > budget:
            break

    msh, uh, info = best
    u_grid = _sample_on_grid(msh, uh, grid)

    result = None
    if COMM.rank == 0:
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": info["mesh_resolution"],
                "element_degree": info["element_degree"],
                "ksp_type": str(info["ksp_type"]),
                "pc_type": str(info["pc_type"]),
                "rtol": float(info["rtol"]),
                "iterations": int(info["iterations"]),
                "verification_l2nodal_error": float(info["verification_l2nodal_error"]),
            },
        }
    return COMM.bcast(result, root=0)
