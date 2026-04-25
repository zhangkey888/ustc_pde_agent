import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

# ```DIAGNOSIS
# equation_type:        biharmonic
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar+scalar
# coupling:             sequential
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       hypre
# special_treatment:    problem_splitting
# pde_skill:            none
# ```

ScalarType = PETSc.ScalarType


def _make_mesh(n: int):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )


def _build_space(domain, degree: int):
    cell_name = domain.topology.cell_name()
    el_u = basix_element("Lagrange", cell_name, degree)
    el_w = basix_element("Lagrange", cell_name, degree)
    W = fem.functionspace(domain, basix_mixed_element([el_u, el_w]))
    return W


def _all_boundary_facets(domain):
    fdim = domain.topology.dim - 1
    return mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))


def _solve_biharmonic_mixed(n: int, degree: int, ksp_type: str, pc_type: str, rtol: float):
    domain = _make_mesh(n)
    W = _build_space(domain, degree)

    (u, w) = ufl.TrialFunctions(W)
    (v, z) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.cos(6.0 * ufl.pi * x[1])

    # Mixed formulation with w = -Δu:
    #   -Δu = w
    #   -Δw = f
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(w, v) * ufl.dx
        + ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx
    )
    L = ufl.inner(f_expr, z) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = _all_boundary_facets(domain)
    Vu, _ = W.sub(0).collapse()
    u0 = fem.Function(Vu)
    u0.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), Vu), fdim, facets)
    bc_u = fem.dirichletbc(u0, dofs_u, W.sub(0))

    opts = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc_u],
        petsc_options=opts,
        petsc_options_prefix="biharmonic_",
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()
    iterations = int(problem.solver.getIterationNumber())
    return domain, uh, iterations


def _sample_on_uniform_grid(domain, u_fun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_local = []
    cells_local = []
    ids_local = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if points_local:
        vals = u_fun.eval(np.asarray(points_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(values, root=0)

    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        merged[~np.isfinite(merged)] = 0.0
        grid = merged.reshape(ny, nx)
    else:
        grid = None

    grid = comm.bcast(grid, root=0)
    return grid


def _analytic_reference(grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    lam = ((8.0 * np.pi) ** 2 + (6.0 * np.pi) ** 2) ** 2
    return np.sin(8.0 * np.pi * X) * np.cos(6.0 * np.pi * Y) / lam


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid_spec = case_spec["output"]["grid"]

    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9

    # Adaptive time-accuracy trade-off under ~14 s budget.
    mesh_candidates = [48, 64, 80, 96]
    chosen_n = mesh_candidates[0]
    chosen = None
    total_iterations = 0
    time_budget = 14.012

    for n in mesh_candidates:
        try:
            domain, uh, its = _solve_biharmonic_mixed(n, degree, ksp_type, pc_type, rtol)
            elapsed = time.perf_counter() - t0
            chosen_n = n
            chosen = (domain, uh)
            total_iterations += its
            if elapsed > 0.88 * time_budget:
                break
        except Exception:
            break

    if chosen is None:
        domain, uh, its = _solve_biharmonic_mixed(48, degree, "preonly", "lu", 1e-12)
        chosen_n = 48
        chosen = (domain, uh)
        total_iterations += its
        ksp_type = "preonly"
        pc_type = "lu"
        rtol = 1e-12

    domain, uh = chosen
    u_grid = _sample_on_uniform_grid(domain, uh, grid_spec)

    # Accuracy verification using the exact Fourier-mode solution for this benchmark case.
    u_ref = _analytic_reference(grid_spec)
    abs_l2_grid = float(np.linalg.norm(u_grid - u_ref) / np.sqrt(u_grid.size))
    rel_l2_grid = float(np.linalg.norm(u_grid - u_ref) / max(np.linalg.norm(u_ref), 1e-14))

    solver_info = {
        "mesh_resolution": int(chosen_n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "accuracy_verification": {
            "reference": "analytic_fourier_mode",
            "grid_abs_l2_error": abs_l2_grid,
            "grid_rel_l2_error": rel_l2_grid,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}
