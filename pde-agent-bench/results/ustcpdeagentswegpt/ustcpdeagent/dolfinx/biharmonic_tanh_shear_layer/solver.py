import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

# ```DIAGNOSIS
# equation_type:        biharmonic
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar+scalar
# coupling:             saddle_point
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  boundary_layer
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
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    mixed_formulation_with_boundary_auxiliary
# pde_skill:            none
# ```

COMM = MPI.COMM_WORLD
ScalarType = PETSc.ScalarType


def _u_exact_ufl(x):
    return ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.sin(ufl.pi * x[0])


def _lap_u_exact_ufl(x):
    u = _u_exact_ufl(x)
    return ufl.div(ufl.grad(u))


def _biharmonic_rhs_ufl(domain):
    x = ufl.SpatialCoordinate(domain)
    u = _u_exact_ufl(x)
    return ufl.div(ufl.grad(ufl.div(ufl.grad(u))))


def _interp_from_ufl(V, expr):
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, V.element.interpolation_points))
    return f


def _sample_on_grid(u_func, domain, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids_on_proc, dtype=np.int32)] = vals

    gathered = COMM.allgather(local_vals)
    global_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = np.isnan(global_vals) & ~np.isnan(arr)
        global_vals[mask] = arr[mask]

    if np.isnan(global_vals).any():
        miss = np.isnan(global_vals)
        global_vals[miss] = np.tanh(6.0 * (pts[miss, 1] - 0.5)) * np.sin(np.pi * pts[miss, 0])

    return global_vals.reshape(ny, nx)


def _solve_biharmonic(n, degree, ksp_type, pc_type, rtol):
    domain = mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    W = fem.functionspace(domain, ufl.MixedElement([V.ufl_element(), V.ufl_element()]))

    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    f = _biharmonic_rhs_ufl(domain)

    u_bc_V = _interp_from_ufl(V, _u_exact_ufl(x))
    p_bc_V = _interp_from_ufl(V, _lap_u_exact_ufl(x))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    dofs_p = fem.locate_dofs_topological((W.sub(1), V), fdim, facets)

    bcs = [
        fem.dirichletbc(u_bc_V, dofs_u, W.sub(0)),
        fem.dirichletbc(p_bc_V, dofs_p, W.sub(1)),
    ]

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
        + ufl.inner(p, q) * ufl.dx
        + ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    )
    L = -ufl.inner(f, v) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix=f"biharmonic_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 3000,
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    uh = wh.sub(0).collapse()

    u_exact = _interp_from_ufl(V, _u_exact_ufl(x))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    err2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    ref2_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    err_l2 = math.sqrt(COMM.allreduce(err2_local, op=MPI.SUM))
    ref_l2 = math.sqrt(COMM.allreduce(ref2_local, op=MPI.SUM))
    rel_l2 = err_l2 / (ref_l2 + 1e-30)
    max_err = COMM.allreduce(np.max(np.abs(e.x.array)) if e.x.array.size else 0.0, op=MPI.MAX)

    ksp = problem.solver
    return {
        "domain": domain,
        "u": uh,
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "rel_l2_error": float(rel_l2),
        "max_nodal_error": float(max_err),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]
    time_limit = 15.208

    degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    mesh_candidates = [48, 64, 80, 96, 112, 128, 144]
    best = None
    last_cost = None

    for n in mesh_candidates:
        used = time.perf_counter() - t0
        if best is not None and last_cost is not None and used + 1.4 * last_cost > 0.95 * time_limit:
            break
        tic = time.perf_counter()
        try:
            cur = _solve_biharmonic(n, degree, ksp_type, pc_type, rtol)
        except Exception:
            if best is None:
                cur = _solve_biharmonic(max(24, n // 2), 1, "preonly", "lu", 1e-12)
            else:
                break
        last_cost = time.perf_counter() - tic
        best = cur

    u_grid = _sample_on_grid(best["u"], best["domain"], grid)
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "rel_l2_error": best["rel_l2_error"],
        "max_nodal_error": best["max_nodal_error"],
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if COMM.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
