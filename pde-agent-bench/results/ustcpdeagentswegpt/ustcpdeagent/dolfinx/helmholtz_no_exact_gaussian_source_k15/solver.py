import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: helmholtz
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: wave
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: helmholtz
"""


def _source_expr(x):
    return 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))


def _build_problem(comm, n, degree, k_value):
    domain = mesh.create_unit_square(comm, nx=n, ny=n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = _source_expr(x)
    k2 = ScalarType(k_value * k_value)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k2 * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    return domain, V, a, L, bc


def _solve_once(comm, n, degree, k_value, rtol=1e-9):
    domain, V, a, L, bc = _build_problem(comm, n, degree, k_value)

    attempts = [
        {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 4000,
            "ksp_gmres_restart": 200,
            "pc_factor_levels": 1,
        },
        {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    ]

    last_err = None
    for i, opts in enumerate(attempts):
        try:
            problem = petsc.LinearProblem(
                a,
                L,
                bcs=[bc],
                petsc_options=opts,
                petsc_options_prefix=f"helmholtz_{n}_{degree}_{i}_",
            )
            uh = problem.solve()
            uh.x.scatter_forward()
            ksp = problem.solver
            reason = int(ksp.getConvergedReason())
            its = int(ksp.getIterationNumber())
            if reason < 0:
                raise RuntimeError(f"KSP failed with reason {reason}")
            return domain, uh, {
                "iterations": its,
                "ksp_type": str(ksp.getType()),
                "pc_type": str(ksp.getPC().getType()),
                "rtol": float(rtol),
            }
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"Helmholtz solve failed on mesh {n}: {last_err}")


def _sample_function(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    indices = []

    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            indices.append(i)

    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.asarray(indices, dtype=np.int32)] = vals

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    values = np.nan_to_num(values, nan=0.0)
    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    out_grid = case_spec["output"]["grid"]

    k_value = float(
        pde.get("wavenumber", pde.get("k", case_spec.get("wavenumber", 15.0)))
    )
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    degree = 2
    rtol = 1e-9

    # Adaptive accuracy/time trade-off:
    # start sufficiently accurate for k=15 and refine while self-verification shows benefit
    # and while runtime remains comfortably below the benchmark cap.
    if max(nx, ny) <= 96:
        candidates = [72, 96, 128]
    elif max(nx, ny) <= 192:
        candidates = [96, 128, 160]
    else:
        candidates = [128, 160]

    # Respect benchmark time budget with margin
    hard_time_limit = 123.384
    budget = min(0.92 * hard_time_limit, 110.0)

    previous_grid = None
    chosen = None
    verification = {"strategy": "mesh_convergence_on_output_grid"}
    total_iterations = 0
    start = time.perf_counter()

    for idx, n in enumerate(candidates):
        domain, uh, info = _solve_once(comm, n, degree, k_value, rtol=rtol)
        total_iterations += int(info["iterations"])
        current_grid = _sample_function(domain, uh, nx, ny, bbox)
        elapsed = time.perf_counter() - start

        if previous_grid is not None:
            denom = max(np.linalg.norm(current_grid.ravel()), 1e-14)
            rel_change = np.linalg.norm((current_grid - previous_grid).ravel()) / denom
            verification.update(
                {
                    "last_relative_grid_change": float(rel_change),
                    "compared_mesh_resolution": int(n),
                }
            )
            chosen = (n, domain, uh, info, current_grid)

            # If already converged or the next refinement may threaten the budget, stop.
            if rel_change < 2.0e-3:
                break
            if elapsed > 0.75 * budget and idx < len(candidates) - 1:
                break
        else:
            chosen = (n, domain, uh, info, current_grid)

        previous_grid = current_grid
        if elapsed > 0.9 * budget:
            break

    n, domain, uh, info, u_grid = chosen
    verification["elapsed_sec"] = float(time.perf_counter() - start)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": float(info["rtol"]),
        "iterations": int(total_iterations),
        "verification": verification,
    }

    return {"u": u_grid, "solver_info": solver_info}
