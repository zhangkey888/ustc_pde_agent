import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _probe_points(u_func, points_array):
    domain = u_func.function_space.mesh
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_array.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return local_vals


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    comm = u_func.function_space.mesh.comm
    local_vals = _probe_points(u_func, pts)

    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        vals = np.nan_to_num(vals, nan=0.0)
        return vals.reshape(ny, nx)
    return None


def _manufactured_exact(x):
    return (1.0 / (0.5 * 13.0 * np.pi**2)) * np.sin(3.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    scalar_type = PETSc.ScalarType

    # Adaptive choice tuned for accuracy within typical time budget
    mesh_resolution = 72
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = fem.Constant(domain, scalar_type(0.5))
    f_expr = ufl.sin(3.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(scalar_type(0.0), boundary_dofs, V)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
            "ksp_max_it": 10000
        }
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification against known analytical solution
    u_exact = fem.Function(V)
    u_exact.interpolate(_manufactured_exact)
    err_fn = fem.Function(V)
    err_fn.x.array[:] = uh.x.array - u_exact.x.array
    l2_error_local = fem.assemble_scalar(fem.form(ufl.inner(err_fn, err_fn) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    wall_time = time.perf_counter() - t0

    iterations = 0
    try:
        ksp = problem.solver
        if ksp is not None:
            iterations = int(ksp.getIterationNumber())
            try:
                ksp_type = ksp.getType()
                pc = ksp.getPC()
                if pc is not None:
                    pc_type = pc.getType()
            except Exception:
                pass
    except Exception:
        iterations = 0

    u_grid = _sample_on_grid(uh, case_spec["output"]["grid"])
    if comm.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": str(ksp_type),
                "pc_type": str(pc_type),
                "rtol": float(rtol),
                "iterations": int(iterations),
                "l2_error_verification": float(l2_error),
                "wall_time_sec": float(wall_time),
            },
        }
    return {
        "u": None,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "l2_error_verification": float(l2_error),
            "wall_time_sec": float(wall_time),
        },
    }
