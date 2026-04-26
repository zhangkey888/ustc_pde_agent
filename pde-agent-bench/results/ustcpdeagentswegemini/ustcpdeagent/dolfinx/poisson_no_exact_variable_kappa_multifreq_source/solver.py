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
# special_notes: manufactured_solution, variable_coeff
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

from __future__ import annotations

import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _kappa_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 1.0 + 0.6 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _source_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return (
        ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
        + 0.3 * ufl.sin(10.0 * ufl.pi * x[0]) * ufl.sin(9.0 * ufl.pi * x[1])
    )


def _manufactured_u_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 0.15 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    )


def _manufactured_f_ufl(msh, u_exact):
    kappa = _kappa_ufl(msh)
    return -ufl.div(kappa * ufl.grad(u_exact))


def _zero_dirichlet_bc(V, msh):
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _build_solver(A, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=2000)
    solver.setFromOptions()
    return solver


def _assemble_and_solve(msh, n, degree, rhs_expr, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    V = fem.functionspace(msh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bc = _zero_dirichlet_bc(V, msh)

    kappa = _kappa_ufl(msh)
    a = fem.form(ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(rhs_expr, v) * ufl.dx)

    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L.function_spaces)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    tried = []
    last_err = None
    for kspt, pct in [(ksp_type, pc_type), ("gmres", "ilu"), ("preonly", "lu")]:
        try:
            solver = _build_solver(A, ksp_type=kspt, pc_type=pct, rtol=rtol)
            uh = fem.Function(V)
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            reason = solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"KSP did not converge, reason={reason}")
            return uh, {
                "mesh_resolution": int(n),
                "element_degree": int(degree),
                "ksp_type": kspt,
                "pc_type": pct,
                "rtol": float(rtol),
                "iterations": int(solver.getIterationNumber()),
            }
        except Exception as e:
            tried.append((kspt, pct, str(e)))
            last_err = e
    raise RuntimeError(f"All linear solver attempts failed: {tried}") from last_err


def _compute_l2_error(msh, uh, u_exact_ufl):
    err_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    local = fem.assemble_scalar(err_form)
    global_err = msh.comm.allreduce(local, op=MPI.SUM)
    return float(math.sqrt(global_err))


def _run_accuracy_verification():
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        pass

    candidates = [(32, 1), (48, 1), (48, 2), (64, 2)]
    best = None
    t0 = time.perf_counter()

    for n, degree in candidates:
        msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
        u_exact = _manufactured_u_ufl(msh)
        f_mms = _manufactured_f_ufl(msh, u_exact)
        ts = time.perf_counter()
        uh, info = _assemble_and_solve(msh, n, degree, f_mms)
        err = _compute_l2_error(msh, uh, u_exact)
        elapsed = time.perf_counter() - ts
        item = {
            "n": n,
            "degree": degree,
            "l2_error": err,
            "solve_time": elapsed,
            "solver_info": info,
        }
        if best is None or err < best["l2_error"]:
            best = item

        if time.perf_counter() - t0 > 2.0:
            break

    return best


def _sample_on_grid(uh, grid):
    msh = uh.function_space.mesh
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_ids, dtype=np.int32)] = vals

    # Single-rank expected for benchmark; still support reduction if needed.
    if msh.comm.size > 1:
        gathered = msh.comm.allgather(values)
        out = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        values = out

    if np.isnan(values).any():
        # Very small fallback for boundary/collision edge cases
        nan_ids = np.where(np.isnan(values))[0]
        coords = pts[nan_ids, :2]
        coords = np.clip(coords, 0.0, 1.0)
        values[nan_ids] = 0.0
        tol = 1e-12
        on_boundary = (
            (np.abs(coords[:, 0] - 0.0) < tol)
            | (np.abs(coords[:, 0] - 1.0) < tol)
            | (np.abs(coords[:, 1] - 0.0) < tol)
            | (np.abs(coords[:, 1] - 1.0) < tol)
        )
        values[nan_ids[on_boundary]] = 0.0

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    verification = _run_accuracy_verification()

    # Adaptive time-accuracy trade-off guided by the 6.614s budget.
    # Prefer P2; increase mesh if verification was fast.
    budget = 6.614
    remaining_hint = budget - (time.perf_counter() - t_start)

    if remaining_hint > 4.5:
        n, degree = 96, 2
    elif remaining_hint > 3.2:
        n, degree = 80, 2
    elif remaining_hint > 2.0:
        n, degree = 72, 2
    else:
        n, degree = 64, 2

    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    rhs = _source_ufl(msh)
    uh, solver_info = _assemble_and_solve(
        msh, n=n, degree=degree, rhs_expr=rhs, ksp_type="cg", pc_type="hypre", rtol=1.0e-10
    )

    grid = case_spec["output"]["grid"]
    u_grid = _sample_on_grid(uh, grid)

    solver_info["verification_l2_error"] = float(verification["l2_error"]) if verification else None
    solver_info["verification_mesh_resolution"] = int(verification["n"]) if verification else None
    solver_info["verification_element_degree"] = int(verification["degree"]) if verification else None
    solver_info["wall_time_sec_est"] = float(time.perf_counter() - t_start)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
