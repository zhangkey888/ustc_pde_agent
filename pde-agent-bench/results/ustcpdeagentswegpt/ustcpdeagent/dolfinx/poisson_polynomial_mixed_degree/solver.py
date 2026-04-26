import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])


def _make_exact_ufl(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])


def _probe_function(u_func, pts):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_ids, local_points, local_cells = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(pts[i])
            local_cells.append(links[0])

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        local_vals[np.array(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(local_vals)
    result = np.full(pts.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        result[mask] = arr[mask]
    return result


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = int(case_spec.get("solver_params", {}).get("mesh_resolution", 18))
    element_degree = int(case_spec.get("solver_params", {}).get("element_degree", 2))
    kappa = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 1.0))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = _make_exact_ufl(x)
    f_expr = -ufl.div(ScalarType(kappa) * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ScalarType(kappa) * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    opts = {
        "ksp_type": case_spec.get("solver_params", {}).get("ksp_type", "cg"),
        "pc_type": case_spec.get("solver_params", {}).get("pc_type", "hypre"),
        "ksp_rtol": float(case_spec.get("solver_params", {}).get("rtol", 1e-10)),
    }

    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts, petsc_options_prefix="poisson_poly_")
        uh = problem.solve()
        ksp = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-12},
            petsc_options_prefix="poisson_poly_fallback_"
        )
        uh = problem.solve()
        ksp = problem.solver

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_sq = comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM)
    l2_error = float(np.sqrt(max(l2_sq, 0.0)))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    vals = _probe_function(uh, pts)
    u_grid = vals.reshape(ny, nx)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "l2_error": l2_error,
        "wall_time_sec": float(time.perf_counter() - t0),
    }

    return {"u": u_grid, "solver_info": solver_info}
