import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: poisson
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: diffusion
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
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType


def _solve_poisson(n, degree, ksp_type, pc_type, rtol, source_kind="benchmark"):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    if source_kind == "benchmark":
        f_expr = ufl.sin(12.0 * ufl.pi * x[0]) * ufl.sin(10.0 * ufl.pi * x[1])
        bc_value = ScalarType(0.0)
        use_function_bc = False
    else:
        u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f_expr = 2.0 * ufl.pi * ufl.pi * u_exact
        use_function_bc = True

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    if use_function_bc:
        u_bc = fem.Function(V)
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, dofs)
    else:
        bc = fem.dirichletbc(bc_value, dofs, V)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{n}_{degree}_{source_kind}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    ksp = problem.solver
    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
    }
    if source_kind == "manufactured":
        return domain, V, uh, info, u_exact
    return domain, V, uh, info


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    local_pts = []
    local_cells = []
    local_ids = []

    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_pts.append(points[i])
            local_cells.append(links[0])
            local_ids.append(i)

    if local_pts:
        vals = uh.eval(np.asarray(local_pts, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_ids), -1)[:, 0]
        values[np.asarray(local_ids, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(values)
    merged = np.full_like(values, np.nan)
    for arr in gathered:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]

    merged = np.nan_to_num(merged, nan=0.0)
    return merged.reshape(ny, nx)


def _verification_error():
    try:
        domain, V, uh, _, u_exact = _solve_poisson(
            n=32, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10, source_kind="manufactured"
        )
        err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
        norm_local = fem.assemble_scalar(fem.form((u_exact) ** 2 * ufl.dx))
        err = domain.comm.allreduce(err_local, op=MPI.SUM)
        norm = domain.comm.allreduce(norm_local, op=MPI.SUM)
        return float(math.sqrt(err / max(norm, 1e-30)))
    except Exception:
        return float("nan")


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    # Use extra resolution/degree for oscillatory forcing while respecting time budget.
    choices = [(96, 2), (128, 2), (96, 3)]
    domain = V = uh = info = None

    for n, degree in choices:
        try:
            domain, V, uh, info = _solve_poisson(
                n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10, source_kind="benchmark"
            )
        except Exception:
            domain, V, uh, info = _solve_poisson(
                n=n, degree=max(1, min(degree, 2)), ksp_type="preonly", pc_type="lu", rtol=1e-10, source_kind="benchmark"
            )
            break
        if time.perf_counter() - t0 > 5.5:
            break

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])
    solver_info = dict(info)
    solver_info["verification_rel_l2"] = _verification_error()
    solver_info["wall_time_sec"] = float(time.perf_counter() - t0)

    return {"u": u_grid, "solver_info": solver_info}
