import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _gaussian_source(x):
    return np.exp(-250.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    if np.isnan(values).any():
        # Should not happen for the unit square grid, but keep robust fallback
        values = np.nan_to_num(values, nan=0.0)

    return values.reshape((ny, nx))


def _solve_once(n, degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    eps = 0.05
    beta_vec = np.array([2.0, 1.0], dtype=np.float64)
    beta = fem.Constant(domain, ScalarType(beta_vec))
    h = ufl.CellDiameter(domain)
    beta_norm = float(np.linalg.norm(beta_vec))
    Peh = beta_norm * h / (2.0 * eps)

    # Robust SUPG tau for convection-diffusion, active in high-Pe regime and harmless otherwise
    tau = h / (2.0 * beta_norm) * (ufl.coth(Peh) - 1.0 / Peh)

    f_expr = ufl.exp(-250.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    a_galerkin = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    )
    L_galerkin = f_expr * v * ufl.dx

    residual_trial = ufl.inner(beta, ufl.grad(u))
    residual_rhs = f_expr
    a_supg = tau * residual_trial * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    L_supg = tau * residual_rhs * ufl.inner(beta, ufl.grad(v)) * ufl.dx

    a = a_galerkin + a_supg
    L = L_galerkin + L_supg

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

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

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
        uh = fem.Function(V)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solve failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1e-12))
        uh = fem.Function(V)
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"

    its = int(solver.getIterationNumber())
    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": its,
    }


def _verification_indicator(grid_spec, degree, base_n, time_limit=126.938):
    t0 = time.time()

    coarse = _solve_once(base_n, degree)
    u_coarse = _sample_function_on_grid(coarse["domain"], coarse["uh"], grid_spec)

    elapsed = time.time() - t0
    refined_n = base_n * 2 if elapsed < 0.35 * time_limit else int(base_n * 1.5)
    refined_n = max(refined_n, base_n + 8)

    fine = _solve_once(refined_n, degree)
    u_fine = _sample_function_on_grid(fine["domain"], fine["uh"], grid_spec)

    diff = u_fine - u_coarse
    linf = float(np.max(np.abs(diff)))
    l2_grid = float(np.sqrt(np.mean(diff ** 2)))

    return fine, {
        "verification_type": "mesh_refinement_grid_difference",
        "coarse_n": int(base_n),
        "fine_n": int(refined_n),
        "linf_diff": linf,
        "l2_grid_diff": l2_grid,
    }, elapsed


def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    time_limit = 126.938
    if "time_limit" in case_spec:
        try:
            time_limit = float(case_spec["time_limit"])
        except Exception:
            pass

    # Start with a robust, accurate P1+SUPG solve on a reasonably fine mesh.
    # If runtime headroom is large, proactively refine to improve accuracy.
    degree = 1
    base_n = 96

    fine, verification, first_stage_time = _verification_indicator(grid_spec, degree, base_n, time_limit=time_limit)

    chosen = fine
    if first_stage_time < 0.15 * time_limit:
        # Plenty of remaining budget: refine once more for higher accuracy.
        try:
            extra_n = min(224, int(1.5 * fine["mesh_resolution"]))
            chosen = _solve_once(extra_n, degree)
            u_prev = _sample_function_on_grid(fine["domain"], fine["uh"], grid_spec)
            u_new = _sample_function_on_grid(chosen["domain"], chosen["uh"], grid_spec)
            verification = {
                "verification_type": "mesh_refinement_grid_difference",
                "coarse_n": int(fine["mesh_resolution"]),
                "fine_n": int(chosen["mesh_resolution"]),
                "linf_diff": float(np.max(np.abs(u_new - u_prev))),
                "l2_grid_diff": float(np.sqrt(np.mean((u_new - u_prev) ** 2))),
            }
        except Exception:
            chosen = fine

    u_grid = _sample_function_on_grid(chosen["domain"], chosen["uh"], grid_spec)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(chosen["element_degree"]),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "stabilization": "SUPG",
        "verification": verification,
        "problem_type": "steady_convection_diffusion",
        "peclet_estimate": 44.7,
    }

    return {"u": u_grid, "solver_info": solver_info}


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
