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

from __future__ import annotations

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc


ScalarType = PETSc.ScalarType


def _source_numpy(x):
    return (
        np.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + np.exp(-250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.70) ** 2))
    )


def _build_and_solve(comm, n: int, degree: int, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f_expr = (
        ufl.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2))
        + ufl.exp(-250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.70) ** 2))
    )

    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f_expr, v) * ufl.dx)

    A = fem_petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = fem_petsc.create_vector(L.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=2000)
    solver.setFromOptions()

    try:
        with b.localForm() as loc:
            loc.set(0)
        fem_petsc.assemble_vector(b, L)
        fem_petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solve failed with reason {reason}")
        its = solver.getIterationNumber()
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-14, atol=0.0, max_it=1)
        solver.setFromOptions()

        with b.localForm() as loc:
            loc.set(0)
        fem_petsc.assemble_vector(b, L)
        fem_petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = 1
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()

    residual_form = fem.form((ufl.inner(ufl.grad(uh), ufl.grad(uh)) - f_expr * uh) * ufl.dx)
    energy = fem.assemble_scalar(residual_form)
    energy = comm.allreduce(energy, op=MPI.SUM)

    return {
        "mesh": msh,
        "V": V,
        "u": uh,
        "iterations": int(its),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "energy_residual": float(abs(energy)),
    }


def _sample_function(msh, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = uh.eval(np.asarray(local_points, dtype=np.float64), np.asarray(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_points), -1)[:, 0]
        local_vals[np.asarray(local_ids, dtype=np.int32)] = vals

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        final = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            final[mask] = arr[mask]
        if np.isnan(final).any():
            final = np.nan_to_num(final, nan=0.0)
        return final.reshape(ny, nx)

    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]

    # Time-budget-aware adaptive accuracy choice.
    # This benchmark allows ~3.8 s, so choose a reasonably accurate default P2 mesh
    # while avoiding over-commitment. If tiny output grid requested, keep same FEM accuracy.
    degree = 2
    mesh_resolution = 72

    result = _build_and_solve(comm, mesh_resolution, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
    u_grid = _sample_function(result["mesh"], result["u"], grid_spec)

    # Accuracy verification without exact solution:
    # compare against one refinement if there is enough budget and running in serial root-visible mode.
    verification = {}
    elapsed = time.perf_counter() - t0
    if elapsed < 2.0:
        try:
            refined_n = min(mesh_resolution + 16, 96)
            result_ref = _build_and_solve(comm, refined_n, degree, ksp_type="cg", pc_type="hypre", rtol=1e-10)
            u_ref_grid = _sample_function(result_ref["mesh"], result_ref["u"], grid_spec)
            if comm.rank == 0:
                diff = np.linalg.norm(u_ref_grid - u_grid) / max(np.linalg.norm(u_ref_grid), 1e-14)
                verification["grid_relative_change_vs_refined"] = float(diff)
                if diff > 2e-3 and elapsed < 3.0:
                    result = result_ref
                    u_grid = u_ref_grid
                    mesh_resolution = refined_n
        except Exception:
            pass

    total_elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": result["ksp_type"],
        "pc_type": result["pc_type"],
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
        "accuracy_verification": {
            "energy_residual_abs": float(result["energy_residual"]),
            **verification,
            "wall_time_sec": float(total_elapsed),
        },
    }

    if comm.rank == 0:
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
