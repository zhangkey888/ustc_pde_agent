import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _probe_points(u_func, points_array):
    """
    points_array: shape (3, N)
    returns shape (N,) for scalar function
    """
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    ptsT = np.ascontiguousarray(points_array.T, dtype=np.float64)
    cell_candidates = geometry.compute_collisions_points(bb_tree, ptsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, ptsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(ptsT[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)
    return values


def _sample_on_uniform_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_points(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = f on unit square with manufactured exact solution
    u = sin(8*pi*x) * sin(pi*y), and sample onto requested grid.
    """
    comm = MPI.COMM_WORLD

    # Fixed manufactured problem data from task
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))
    if abs(kappa) < 1e-14:
        raise ValueError("kappa must be nonzero")

    # Choose accuracy/time-balanced discretization.
    # High frequency in x => use moderately fine mesh and P2 for strong accuracy under tight runtime.
    mesh_resolution = 40
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = kappa * (64.0 * ufl.pi**2 + 1.0 * ufl.pi**2) * ufl.sin(8.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Boundary condition from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
            "ksp_norm_type": "unpreconditioned",
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    # Accuracy verification: relative L2 error against manufactured exact solution
    u_exact_fn = fem.Function(V)
    u_exact_fn.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_fn = fem.Function(V)
    err_fn.x.array[:] = uh.x.array - u_exact_fn.x.array
    err_fn.x.scatter_forward()

    local_err2 = fem.assemble_scalar(fem.form(ufl.inner(err_fn, err_fn) * ufl.dx))
    local_ref2 = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fn, u_exact_fn) * ufl.dx))
    global_err2 = comm.allreduce(local_err2, op=MPI.SUM)
    global_ref2 = comm.allreduce(local_ref2, op=MPI.SUM)
    rel_l2_error = float(np.sqrt(global_err2 / global_ref2)) if global_ref2 > 0 else float(np.sqrt(global_err2))

    # Try to report iterations if available
    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        try:
            iterations = int(problem.solver.getIterationNumber)
        except Exception:
            iterations = 0

    u_grid = _sample_on_uniform_grid(uh, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": iterations,
        "verification": {
            "relative_l2_error": rel_l2_error,
            "solve_wall_time_sec": float(solve_time),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"coefficients": {"kappa": 1.0}},
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    assert isinstance(out, dict)
    assert out["u"].shape == (24, 32)
    assert np.isfinite(out["u"]).all()
    print("OK", out["u"].shape, out["solver_info"])
