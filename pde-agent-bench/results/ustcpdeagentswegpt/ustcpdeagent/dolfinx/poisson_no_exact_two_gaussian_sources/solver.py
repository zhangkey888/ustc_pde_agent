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


def _source_expr(x):
    return np.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2)) + np.exp(
        -250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.7) ** 2)
    )


def _build_and_solve(n, degree=2, ksp_type="cg", pc_type="hypre", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, nx=n, ny=n, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_ufl = ufl.exp(-250.0 * ((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2)) + ufl.exp(
        -250.0 * ((x[0] - 0.75) ** 2 + (x[1] - 0.7) ** 2)
    )
    kappa = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

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
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solver failed, reason={reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-14, atol=0.0, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    its = int(solver.getIterationNumber())
    return domain, V, uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(its),
    }


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        merged = np.full_like(values, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        values = merged

    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=0.0)

    return values.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid = case_spec["output"]["grid"]

    t0 = time.perf_counter()

    # Start with a reasonably accurate choice; adapt upward if time allows.
    candidate_resolutions = [48, 64, 80]
    degree = 2
    chosen = None
    chosen_grid = None
    refined_grid = None
    final_info = None

    for idx, n in enumerate(candidate_resolutions):
        domain, V, uh, info = _build_and_solve(
            n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10
        )
        u_grid = _sample_on_grid(domain, uh, grid)

        elapsed = time.perf_counter() - t0
        remaining_budget = 3.826 - elapsed

        # Accuracy verification module: compare with one refined solve on same output grid
        # if enough time remains; use this as a self-convergence indicator.
        err_est = None
        if remaining_budget > 0.8:
            n_ref = min(int(1.5 * n), max(n + 8, n * 2 // 1))
            domain_r, V_r, uh_r, info_r = _build_and_solve(
                n=n_ref, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1e-10
            )
            u_ref_grid = _sample_on_grid(domain_r, uh_r, grid)
            err_est = float(np.sqrt(np.mean((u_ref_grid - u_grid) ** 2)))
            refined_grid = u_ref_grid
            # If we still have substantial time, continue refining candidate choice.
            chosen = (domain, uh)
            chosen_grid = u_grid
            final_info = info
            final_info["iterations"] += info_r["iterations"]
            final_info["self_convergence_l2_grid"] = err_est
            if idx < len(candidate_resolutions) - 1 and (time.perf_counter() - t0) < 2.6:
                continue
            # Prefer refined solution if computed and affordable
            chosen = (domain_r, uh_r)
            chosen_grid = u_ref_grid
            final_info.update(
                {
                    "mesh_resolution": int(n_ref),
                    "element_degree": int(degree),
                    "ksp_type": info_r["ksp_type"],
                    "pc_type": info_r["pc_type"],
                    "rtol": info_r["rtol"],
                }
            )
            break
        else:
            chosen = (domain, uh)
            chosen_grid = u_grid
            final_info = info
            final_info["self_convergence_l2_grid"] = None
            break

    if chosen_grid is None:
        domain, V, uh, info = _build_and_solve(n=64, degree=degree)
        chosen_grid = _sample_on_grid(domain, uh, grid)
        final_info = info
        final_info["self_convergence_l2_grid"] = None

    result = {
        "u": np.asarray(chosen_grid, dtype=np.float64),
        "solver_info": final_info,
    }
    return result
