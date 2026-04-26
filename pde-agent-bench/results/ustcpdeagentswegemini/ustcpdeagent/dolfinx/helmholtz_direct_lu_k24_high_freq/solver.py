import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: helmholtz
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: wave
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: helmholtz
# ```

ScalarType = PETSc.ScalarType


def _manufactured_ufl(msh, k: float):
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    # For -Δu - k^2 u = f and Δu = -(41*pi^2)u:
    # f = (41*pi^2 - k^2) u
    f = (41.0 * ufl.pi**2 - k**2) * u_exact
    return u_exact, f


def _build_problem(n: int, degree: int, k: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u_exact_ufl, f_ufl = _manufactured_ufl(msh, k)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Weak form for -Δu - k^2 u = f:
    # ∫ grad(u).grad(v) dx - k^2 ∫ u v dx = ∫ f v dx
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=f"helmholtz_{n}_{degree}_",
    )
    return msh, V, u_exact_ufl, problem


def _compute_l2_error(msh, uh, u_exact_ufl):
    diff = uh - u_exact_ufl
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx))
    return float(np.sqrt(msh.comm.allreduce(l2_local, op=MPI.SUM)))


def _sample_on_grid(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]

    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
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
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(idxs, dtype=np.int32)] = np.real(vals).reshape(-1)

    gathered = msh.comm.gather(local_vals, root=0)
    if msh.comm.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to sample solution at some output grid points.")
        return merged.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    k = float(case_spec.get("pde", {}).get("k", 24.0))
    grid = case_spec["output"]["grid"]

    degree = 2
    n = 96 if MPI.COMM_WORLD.size == 1 else 72

    msh, V, u_exact_ufl, problem = _build_problem(n, degree, k)
    uh = problem.solve()
    uh.x.scatter_forward()

    l2_error = _compute_l2_error(msh, uh, u_exact_ufl)
    elapsed = time.perf_counter() - t0

    # Adaptive time-accuracy tradeoff: if initial solve is comfortably fast, refine once.
    if elapsed < 1.5:
        n_try = 128 if MPI.COMM_WORLD.size == 1 else 96
        msh2, V2, u_exact_ufl2, problem2 = _build_problem(n_try, degree, k)
        uh2 = problem2.solve()
        uh2.x.scatter_forward()
        l2_error2 = _compute_l2_error(msh2, uh2, u_exact_ufl2)
        elapsed2 = time.perf_counter() - t0
        if elapsed2 < 2.4 and l2_error2 <= l2_error:
            msh, V, u_exact_ufl, problem, uh, l2_error, n = (
                msh2,
                V2,
                u_exact_ufl2,
                problem2,
                uh2,
                l2_error2,
                n_try,
            )

    u_grid = _sample_on_grid(msh, uh, grid)

    ksp = problem.solver
    ksp_type = str(ksp.getType()).lower()
    pc_type = str(ksp.getPC().getType()).lower()
    rtol = 0.0 if ksp_type == "preonly" else float(ksp.getTolerances()[0])

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(ksp.getIterationNumber()),
    }

    if msh.comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 24.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0

    if MPI.COMM_WORLD.rank == 0:
        nx = case_spec["output"]["grid"]["nx"]
        ny = case_spec["output"]["grid"]["ny"]
        bbox = case_spec["output"]["grid"]["bbox"]
        xs = np.linspace(bbox[0], bbox[1], nx)
        ys = np.linspace(bbox[2], bbox[3], ny)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        u_exact_grid = np.sin(5.0 * np.pi * XX) * np.sin(4.0 * np.pi * YY)
        l2_grid = float(np.sqrt(np.mean((out["u"] - u_exact_grid) ** 2)))
        print(f"L2_ERROR: {l2_grid:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(out["u"].shape)
        print(out["solver_info"])
