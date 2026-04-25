import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


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
# special_notes: variable_coeff
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: amg
# special_treatment: none
# pde_skill: poisson
# ```


def _kappa_numpy(x, y):
    return 1.0 + 0.6 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def _f_numpy(x, y):
    return (
        np.sin(4.0 * np.pi * x) * np.sin(3.0 * np.pi * y)
        + 0.3 * np.sin(10.0 * np.pi * x) * np.sin(9.0 * np.pi * y)
    )


def _build_problem(comm, n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa_expr = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )

    a = ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1e-14,
    }
    if ksp_type == "cg" and pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    uh = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options=opts,
        petsc_options_prefix=f"poisson_{n}_{degree}_",
    ).solve()
    uh.x.scatter_forward()

    ksp = PETSc.KSP().create(domain.comm)
    A = petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    b = petsc.create_vector(fem.form(L).function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, fem.form(L))
    petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    try:
        if pc_type == "hypre":
            ksp.getPC().setHYPREType("boomeramg")
    except Exception:
        pass
    ksp.setTolerances(rtol=rtol, atol=1e-14)
    tmp = fem.Function(V)
    ksp.solve(b, tmp.x.petsc_vec)
    tmp.x.scatter_forward()
    iterations = int(ksp.getIterationNumber())

    return domain, V, uh, iterations


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    comm = domain.comm
    full = np.empty_like(values)
    comm.Allreduce(values, full, op=MPI.SUM)

    if np.isnan(full).any():
        # Handle boundary/corner misses robustly with tiny inward shift
        miss = np.isnan(full)
        pts2 = pts.copy()
        eps = 1e-12
        pts2[miss, 0] = np.minimum(np.maximum(pts2[miss, 0], 0.0 + eps), 1.0 - eps)
        pts2[miss, 1] = np.minimum(np.maximum(pts2[miss, 1], 0.0 + eps), 1.0 - eps)
        cell_candidates = geometry.compute_collisions_points(tree, pts2)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)
        points_on_proc = []
        cells = []
        eval_map = []
        for idx in np.where(miss)[0]:
            links = colliding_cells.links(idx)
            if len(links) > 0:
                points_on_proc.append(pts2[idx])
                cells.append(links[0])
                eval_map.append(idx)
        values2 = np.zeros_like(values)
        if points_on_proc:
            vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
            vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
            values2[np.array(eval_map, dtype=np.int32)] = vals
        add = np.empty_like(values2)
        comm.Allreduce(values2, add, op=MPI.SUM)
        full[miss] = add[miss]

    return full.reshape((ny, nx))


def _estimate_accuracy(domain, uh):
    V0 = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(V0)
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )
    residual_indicator = fem.assemble_scalar(
        fem.form((kappa * ufl.inner(ufl.grad(uh), ufl.grad(uh)) - 2.0 * f_expr * uh) * ufl.dx)
    )
    residual_indicator = float(domain.comm.allreduce(residual_indicator, op=MPI.SUM))

    # Reference grid comparison by bilinear-style point sampling delta is handled externally.
    return {"energy_like": residual_indicator}


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    budget = 6.774
    reserve = 0.9

    candidates = [
        (64, 2, "cg", "hypre", 1e-10),
        (80, 2, "cg", "hypre", 1e-10),
        (96, 2, "cg", "hypre", 1e-10),
        (112, 2, "cg", "hypre", 1e-10),
    ]

    chosen = None
    chosen_grid = None
    chosen_info = None
    prev_grid = None
    prev_meta = None

    for n, degree, ksp_type, pc_type, rtol in candidates:
        now = time.perf_counter()
        if now - t0 > max(0.5, budget - reserve):
            break

        try:
            domain, V, uh, iterations = _build_problem(comm, n, degree, ksp_type, pc_type, rtol)
        except Exception:
            if ksp_type == "cg":
                domain, V, uh, iterations = _build_problem(comm, n, degree, "preonly", "lu", rtol)
                ksp_type, pc_type = "preonly", "lu"
            else:
                raise

        u_grid = _sample_on_grid(domain, uh, output_grid)
        acc = _estimate_accuracy(domain, uh)

        delta = None
        if prev_grid is not None and prev_grid.shape == u_grid.shape:
            delta = float(np.linalg.norm(u_grid - prev_grid) / math.sqrt(u_grid.size))

        chosen = uh
        chosen_grid = u_grid
        chosen_info = {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "verification": {
                "energy_like": acc["energy_like"],
                "grid_change_l2": delta,
            },
        }

        # Adaptive accuracy/time trade-off:
        # if solve is comfortably within budget, continue refining.
        elapsed = time.perf_counter() - t0
        prev_grid = u_grid
        prev_meta = (n, degree)

        # Stop if convergence in sampled grid is already strong and little time remains
        if delta is not None and delta < 3e-3 and elapsed > 0.45 * budget:
            break
        if elapsed > budget - reserve:
            break

    if chosen_grid is None:
        raise RuntimeError("Poisson solve failed for all candidate discretizations.")

    return {
        "u": chosen_grid,
        "solver_info": chosen_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
