import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _make_kappa_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))


def _make_exact_expr(domain):
    x = ufl.SpatialCoordinate(domain)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _probe_points(u_func, points_array):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    pts = np.asarray(points_array, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_array must have shape (N, 3)")
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals_local = _probe_points(u_func, points)

    comm = u_func.function_space.mesh.comm
    vals_global = np.empty_like(vals_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.MAX)

    if np.isnan(vals_global).any():
        nan_mask = np.isnan(vals_global)
        vals_global[nan_mask] = 0.0

    return vals_global.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": sampled solution on requested uniform grid, shape (ny, nx)
    - "solver_info": metadata about discretization and solver
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    grid_spec = case_spec["output"]["grid"]

    # Conservative defaults chosen to comfortably satisfy error target under tight wall-time.
    mesh_resolution = 40
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    u_exact_expr = _make_exact_expr(domain)
    kappa_expr = _make_kappa_expr(domain)
    f_expr = -ufl.div(kappa_expr * ufl.grad(u_exact_expr))

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_norm_type": "unpreconditioned",
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Record actual KSP behavior
    ksp = problem.solver
    try:
        iterations = int(ksp.getIterationNumber())
    except Exception:
        iterations = -1

    # Accuracy verification against manufactured exact solution
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_ex.x.array
    e.x.scatter_forward()

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    h1_semi_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    h1_semi_error = math.sqrt(comm.allreduce(h1_semi_local, op=MPI.SUM))

    u_grid = _sample_on_uniform_grid(uh, grid_spec)

    solve_wall_time = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification_l2_error": float(l2_error),
        "verification_h1_semi_error": float(h1_semi_error),
        "wall_time_sec": float(solve_wall_time),
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
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
