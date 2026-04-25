import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _gaussian_source(x):
    return 10.0 * np.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))


def _build_problem(mesh_resolution: int, element_degree: int):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    x = ufl.SpatialCoordinate(domain)
    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))
    return domain, V, bc, f_expr


def _solve_poisson(domain, V, rhs, bc, prefix: str, ksp_type: str, pc_type: str, rtol: float):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rhs * v * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-14,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    its = int(problem.solver.getIterationNumber())
    return uh, its


def _sample_function_on_grid(domain, u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idxs = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.asarray(idxs, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    gathered = domain.comm.allgather(local_vals)
    global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_vals[mask] = arr[mask]
    global_vals[np.isnan(global_vals)] = 0.0
    return global_vals.reshape((ny, nx))


def _solve_biharmonic(mesh_resolution: int, element_degree: int, ksp_type: str, pc_type: str, rtol: float):
    domain, V, bc, f_expr = _build_problem(mesh_resolution, element_degree)

    # Mixed-split surrogate: two Poisson solves with homogeneous Dirichlet conditions.
    # v = -Δu,  -Δv = f
    v_h, it1 = _solve_poisson(domain, V, f_expr, bc, "biharm_v_", ksp_type, pc_type, rtol)
    u_h, it2 = _solve_poisson(domain, V, v_h, bc, "biharm_u_", ksp_type, pc_type, rtol)

    return domain, u_h, int(it1 + it2)


def solve(case_spec: dict) -> dict:
    # Adaptive choice tuned to use time budget while preserving margin.
    mesh_resolution = 320
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain, u_h, iterations = _solve_biharmonic(
        mesh_resolution, element_degree, ksp_type, pc_type, rtol
    )

    u_grid = _sample_function_on_grid(domain, u_h, case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    return {"u": u_grid, "solver_info": solver_info}


def _run_self_test():
    comm = MPI.COMM_WORLD
    case_spec = {
        "pde": {"time": None},
        "output": {
            "grid": {
                "nx": 128,
                "ny": 128,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }

    t0 = time.perf_counter()
    out = solve(case_spec)
    t1 = time.perf_counter()

    # Accuracy verification by self-convergence on the output grid:
    # compare production settings against a finer reference solve.
    ref_domain, ref_u, _ = _solve_biharmonic(
        mesh_resolution=384,
        element_degree=2,
        ksp_type="cg",
        pc_type="hypre",
        rtol=1.0e-11,
    )
    ref_grid = _sample_function_on_grid(ref_domain, ref_u, case_spec["output"]["grid"])
    err = np.sqrt(np.mean((out["u"] - ref_grid) ** 2))

    if comm.rank == 0:
        print(f"L2_ERROR: {err:.12e}")
        print(f"WALL_TIME: {t1 - t0:.12e}")
        print(out["u"].shape)
        print(out["solver_info"])


if __name__ == "__main__":
    _run_self_test()
