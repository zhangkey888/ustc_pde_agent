import time
from typing import Dict, Tuple, List

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

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
# preconditioner: hypre
# special_treatment: none
# pde_skill: poisson
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _make_mesh(n: int):
    return mesh.create_unit_square(COMM, n, n, cell_type=mesh.CellType.triangle)


def _kappa_expr(x):
    return 1.0 + 50.0 * np.exp(-200.0 * (x[0] - 0.5) ** 2)


def _f_expr(x):
    return 1.0 + np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])


def _build_and_solve(n: int, degree: int, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    msh = _make_mesh(n)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5) ** 2)
    f = 1.0 + ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    # robustness fallback
    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP did not converge, reason={reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(msh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    its = int(solver.getIterationNumber())
    return msh, V, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": its,
    }


def _sample_on_grid(msh, uh: fem.Function, grid_spec: Dict) -> np.ndarray:
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts2)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts2[i])
            local_cells.append(links[0])
            local_ids.append(i)

    vals = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
    if local_points:
        eval_vals = uh.eval(np.asarray(local_points, dtype=np.float64),
                            np.asarray(local_cells, dtype=np.int32))
        vals[np.asarray(local_ids, dtype=np.int32)] = np.asarray(eval_vals).reshape(-1)

    if msh.comm.size > 1:
        recv = np.empty_like(vals)
        msh.comm.Allreduce(vals, recv, op=MPI.SUM)
        vals = recv

        mask_local = np.where(~np.isnan(vals), 1, 0).astype(np.int32)
        mask_global = np.empty_like(mask_local)
        msh.comm.Allreduce(mask_local, mask_global, op=MPI.SUM)
        vals = np.where(mask_global > 0, vals, 0.0)
    else:
        vals = np.nan_to_num(vals, nan=0.0)

    return vals.reshape((ny, nx))


def _l2_difference_coarse_fine(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    diff = grid_a - grid_b
    return float(np.sqrt(np.mean(diff * diff)))


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid_spec = case_spec["output"]["grid"]

    # Candidate set chosen to stay under the time budget while exploiting it for accuracy.
    candidates: List[Tuple[int, int]] = [(56, 1), (64, 1), (72, 1), (48, 2), (56, 2), (64, 2), (72, 2), (80, 2), (96, 2), (112, 2)]
    target_budget = 3.75  # leave some margin under 4.004s
    best = None
    best_grid = None
    verification = {}

    prev_grid = None
    prev_meta = None

    for n, degree in candidates:
        now = time.perf_counter()
        if now - t0 > target_budget:
            break

        msh, V, uh, meta = _build_and_solve(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        u_grid = _sample_on_grid(msh, uh, grid_spec)

        # Accuracy verification without exact solution:
        # compare against previous discretization on same output grid.
        if prev_grid is not None:
            err_est = _l2_difference_coarse_fine(u_grid, prev_grid)
            verification = {
                "verification_type": "grid_comparison",
                "estimated_l2_diff": float(err_est),
                "compared_to": {
                    "mesh_resolution": prev_meta["mesh_resolution"],
                    "element_degree": prev_meta["element_degree"],
                },
            }
        else:
            verification = {
                "verification_type": "grid_comparison",
                "estimated_l2_diff": None,
                "compared_to": None,
            }

        best = (msh, V, uh, meta)
        best_grid = u_grid
        prev_grid = u_grid
        prev_meta = meta

        # If we already have a very small change and limited remaining time, stop.
        elapsed = time.perf_counter() - t0
        if verification["estimated_l2_diff"] is not None and verification["estimated_l2_diff"] < 2e-3:
            if elapsed > 1.5:
                break

    # Fallback if loop somehow did not run
    if best is None:
        msh, V, uh, meta = _build_and_solve(n=48, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10)
        best = (msh, V, uh, meta)
        best_grid = _sample_on_grid(msh, uh, grid_spec)
        verification = {
            "verification_type": "grid_comparison",
            "estimated_l2_diff": None,
            "compared_to": None,
        }

    _, _, _, meta = best
    solver_info = dict(meta)
    solver_info.update(verification)
    solver_info["wall_time_sec"] = float(time.perf_counter() - t0)

    return {"u": np.asarray(best_grid, dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
