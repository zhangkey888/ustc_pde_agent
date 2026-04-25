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
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: helmholtz
# ```

ScalarType = PETSc.ScalarType


def _build_and_solve(comm, n, degree, k_value, ksp_type="gmres", pc_type="ilu", rtol=1e-8):
    domain = mesh.create_unit_square(comm, nx=n, ny=n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = 10.0 * ufl.exp(-80.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.55) ** 2))
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k_value * k_value) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    options = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if ksp_type == "gmres" and pc_type == "ilu":
        options.update({"ksp_max_it": 5000, "pc_factor_levels": 0, "ksp_gmres_restart": 200})
    if ksp_type == "preonly" and pc_type == "lu":
        options.update({"pc_factor_mat_solver_type": "mumps"})

    try:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options=options,
            petsc_options_prefix=f"helmholtz_{n}_{degree}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        reason = int(ksp.getConvergedReason())
        if reason < 0:
            raise RuntimeError(f"KSP failed: {reason}")
        return domain, uh, {
            "iterations": its,
            "ksp_type": str(ksp.getType()),
            "pc_type": str(ksp.getPC().getType()),
            "rtol": float(rtol),
        }
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix=f"helmholtz_lu_{n}_{degree}_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        return domain, uh, {"iterations": 1, "ksp_type": "preonly", "pc_type": "lu", "rtol": float(rtol)}


def _sample_function(domain, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

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
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(indices, dtype=np.int32)] = vals

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
    k_value = float(case_spec.get("pde", {}).get("wavenumber", 15.0))
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    degree = 2
    wall_limit = 34.229
    start = time.perf_counter()

    candidates = [64, 96]
    if max(nx, ny) > 160:
        candidates.append(128)

    previous_grid = None
    verification = {}
    chosen = None

    for n in candidates:
        domain, uh, info = _build_and_solve(MPI.COMM_WORLD, n, degree, k_value, "gmres", "ilu", 1e-8)
        current_grid = _sample_function(domain, uh, nx, ny, bbox)
        elapsed = time.perf_counter() - start

        if previous_grid is not None:
            rel_change = np.linalg.norm(current_grid - previous_grid) / max(np.linalg.norm(current_grid), 1e-14)
            verification = {"relative_grid_change": float(rel_change), "compared_mesh_resolution": int(n)}
            chosen = (n, domain, uh, info, current_grid)
            if rel_change < 5e-3 or elapsed > 0.7 * wall_limit:
                break
        else:
            chosen = (n, domain, uh, info, current_grid)

        previous_grid = current_grid
        if elapsed > 0.85 * wall_limit:
            break

    n, domain, uh, info, u_grid = chosen
    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": info["ksp_type"],
        "pc_type": info["pc_type"],
        "rtol": float(info["rtol"]),
        "iterations": int(info["iterations"]),
        "verification": verification,
    }
    return {"u": u_grid, "solver_info": solver_info}
