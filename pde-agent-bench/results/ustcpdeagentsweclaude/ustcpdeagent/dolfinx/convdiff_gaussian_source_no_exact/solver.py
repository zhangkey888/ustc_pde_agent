import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

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

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            merged[mask] = arr[mask]
        if np.any(~np.isfinite(merged)):
            merged[~np.isfinite(merged)] = 0.0
        return merged.reshape((ny, nx))
    return None


def _compute_streamline_residual_indicator(domain, uh, eps_value, beta_vec, f_expr, tau_expr):
    x = ufl.SpatialCoordinate(domain)
    beta = ufl.as_vector(beta_vec)
    r = -eps_value * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - f_expr
    indicator_form = fem.form((tau_expr * r * r) * ufl.dx)
    local_val = fem.assemble_scalar(indicator_form)
    return domain.comm.allreduce(local_val, op=MPI.SUM)


def _build_and_solve(domain, n, degree, ksp_type, pc_type, rtol):
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps_value = 0.02
    beta_vec = (8.0, 3.0)
    beta = ufl.as_vector(beta_vec)
    beta_norm = float(np.sqrt(beta_vec[0] ** 2 + beta_vec[1] ** 2))
    h = ufl.CellDiameter(domain)

    f_expr = ufl.exp(-250.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))

    # SUPG parameter for convection-dominated regime
    tau = h / (2.0 * beta_norm + 4.0 * eps_value / h)

    a = (
        eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * f_expr * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    )

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    if ksp_type.lower() == "gmres":
        solver.setGMRESRestart(200)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        converged = solver.getConvergedReason() > 0
    except Exception:
        converged = False

    if not converged:
        solver.destroy()
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        used_ksp = "preonly"
        used_pc = "lu"
    else:
        used_ksp = solver.getType()
        used_pc = solver.getPC().getType()

    iterations = int(max(solver.getIterationNumber(), 1))
    residual_indicator = _compute_streamline_residual_indicator(domain, uh, eps_value, beta_vec, f_expr, tau)

    return uh, {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": iterations,
        "residual_indicator": float(residual_indicator),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid_spec = case_spec["output"]["grid"]

    # Accuracy/time-aware default selection for this convection-dominated elliptic problem
    diffusion = 0.02
    beta = np.array([8.0, 3.0], dtype=np.float64)
    peclet_est = np.linalg.norm(beta) / diffusion

    if peclet_est > 100:
        degree = 1
        n_candidates = [96, 128, 160]
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-9
    else:
        degree = 2
        n_candidates = [64, 96, 128]
        ksp_type = "cg"
        pc_type = "hypre"
        rtol = 1e-10

    chosen = None
    best_indicator = np.inf

    for idx, n in enumerate(n_candidates):
        domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
        uh, info = _build_and_solve(domain, n, degree, ksp_type, pc_type, rtol)
        elapsed = time.perf_counter() - t0
        indicator = info["residual_indicator"]

        chosen = (domain, uh, info)
        best_indicator = indicator

        # If we are still well below the time budget, continue refining.
        if elapsed < 0.55 * 42.044 and idx < len(n_candidates) - 1:
            continue
        # If indicator is already small enough, accept.
        if indicator < 1e-5:
            break

    domain, uh, info = chosen
    u_grid = _sample_function_on_grid(domain, uh, grid_spec)

    if comm.rank == 0:
        out = {
            "u": np.asarray(u_grid, dtype=np.float64),
            "solver_info": {
                "mesh_resolution": info["mesh_resolution"],
                "element_degree": info["element_degree"],
                "ksp_type": info["ksp_type"],
                "pc_type": info["pc_type"],
                "rtol": info["rtol"],
                "iterations": info["iterations"],
                "residual_indicator": best_indicator,
                "wall_time_sec": time.perf_counter() - t0,
            },
        }
        return out
    return {"u": None, "solver_info": info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
