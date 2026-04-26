import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _defaults(case_spec):
    case_spec = {} if case_spec is None else dict(case_spec)
    case_spec.setdefault("output", {})
    case_spec["output"].setdefault("grid", {})
    g = case_spec["output"]["grid"]
    g.setdefault("nx", 129)
    g.setdefault("ny", 129)
    g.setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    return case_spec


def _boundary_value(x):
    vals = np.zeros((2, x.shape[1]), dtype=np.float64)
    vals[0, :] = np.sin(np.pi * x[1])
    return vals


def _sample_magnitude(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    out = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells_on_proc, dtype=np.int32))
        out[np.asarray(ids, dtype=np.int32)] = np.linalg.norm(vals, axis=1)

    gathered = domain.comm.gather(out, root=0)
    if domain.comm.rank == 0:
        final = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            raise RuntimeError("Point evaluation failed for some output grid points.")
        final = final.reshape((ny, nx))
    else:
        final = None
    return domain.comm.bcast(final, root=0)


def _build_and_solve(n, degree, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(gdim)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, np.array((0.0, 0.0), dtype=ScalarType))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_boundary_value)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    solver_configs = [
        {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 20000,
        },
        {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 20000,
        },
        {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    ]

    last_error = None
    for i, opts in enumerate(solver_configs):
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options_prefix=f"elasticity_{i}_",
                petsc_options=opts,
            )
            uh = problem.solve()
            uh.x.scatter_forward()
            ksp = problem.solver
            reason = ksp.getConvergedReason()
            if reason <= 0 and opts["ksp_type"] != "preonly":
                raise RuntimeError(f"KSP failed with reason {reason}")
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"All solver strategies failed: {last_error}")

    return {
        "domain": domain,
        "solution": uh,
        "iterations": int(ksp.getIterationNumber()),
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "rtol": float(rtol),
    }


def solve(case_spec: dict) -> dict:
    case_spec = _defaults(case_spec)
    t0 = time.perf_counter()

    time_limit = 19.775
    safety = 2.0
    target_budget = max(2.0, time_limit - safety)

    degree = 2
    rtol = 1.0e-10

    candidate_meshes = [48, 72, 96, 128, 160]
    chosen = None
    previous_grid = None
    previous_result = None
    verification = {}

    for n in candidate_meshes:
        stage_t0 = time.perf_counter()
        result = _build_and_solve(n, degree, rtol)
        grid = _sample_magnitude(result["domain"], result["solution"], case_spec["output"]["grid"])
        stage_time = time.perf_counter() - stage_t0

        if previous_grid is not None:
            diff = grid - previous_grid
            verification["grid_convergence_l2"] = float(np.sqrt(np.mean(diff * diff)))
            verification["previous_mesh_resolution"] = int(previous_result["mesh_resolution"])
        else:
            verification["grid_convergence_l2"] = None
            verification["previous_mesh_resolution"] = None

        chosen = (result, grid, stage_time, dict(verification))
        previous_grid = grid
        previous_result = result

        elapsed = time.perf_counter() - t0
        next_index = candidate_meshes.index(n) + 1
        if next_index < len(candidate_meshes):
            projected = elapsed + stage_time
            if projected > target_budget:
                break

    result, u_grid, solve_stage_time, verification = chosen
    total_wall = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": result["mesh_resolution"],
        "element_degree": result["element_degree"],
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": result["rtol"],
        "iterations": result["iterations"],
                "wall_time_check": float(total_wall),
        "grid_convergence_l2": verification["grid_convergence_l2"],
        "previous_mesh_resolution": verification["previous_mesh_resolution"],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 129, "ny": 129, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
