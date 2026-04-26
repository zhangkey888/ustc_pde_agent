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


def _boundary_value(x):
    return np.sin(2.0 * np.pi * x[0]) + 0.5 * np.cos(2.0 * np.pi * x[1])


def _source_value(x):
    return np.exp(-180.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))


def _build_and_solve(comm, n, degree=2, rtol=1e-10, ksp_type="cg", pc_type="hypre"):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = fem.Constant(domain, ScalarType(1.0))
    f_expr = ufl.exp(-180.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
    g_expr = ufl.sin(2.0 * ufl.pi * x[0]) + 0.5 * ufl.cos(2.0 * ufl.pi * x[1])

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

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

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if ksp_type in ("cg", "gmres", "minres", PETSc.KSP.Type.CG, PETSc.KSP.Type.GMRES, PETSc.KSP.Type.MINRES):
        solver.setTolerances(rtol=rtol, atol=1e-14, max_it=2000)
    else:
        solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1)
    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"KSP diverged with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    return domain, V, uh, solver


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    if domain.comm.size > 1:
        recv = np.empty_like(values)
        domain.comm.Allreduce(values, recv, op=MPI.SUM)
        values = recv

    values = values.reshape((ny, nx))

    if np.isnan(values).any():
        # For boundary/partition corner robustness, fill with exact Dirichlet data where needed
        mask = np.isnan(values)
        values[mask] = np.sin(2.0 * np.pi * XX[mask]) + 0.5 * np.cos(2.0 * np.pi * YY[mask])

    return values


def _accuracy_verification(domain, uh, grid):
    nx = max(int(grid["nx"]), 33)
    ny = max(int(grid["ny"]), 33)
    bbox = grid["bbox"]
    probe_grid = {"nx": nx, "ny": ny, "bbox": bbox}
    vals = _sample_on_grid(domain, uh, probe_grid)

    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    g = np.sin(2.0 * np.pi * XX) + 0.5 * np.cos(2.0 * np.pi * YY)

    bmask = (
        np.isclose(XX, bbox[0]) | np.isclose(XX, bbox[1]) |
        np.isclose(YY, bbox[2]) | np.isclose(YY, bbox[3])
    )
    bc_max_err = float(np.max(np.abs(vals[bmask] - g[bmask])))

    center_val = float(_sample_on_grid(domain, uh, {"nx": 1, "ny": 1, "bbox": [0.5, 0.5, 0.5, 0.5]})[0, 0])

    return {
        "boundary_max_error": bc_max_err,
        "probe_center": center_val,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    grid = case_spec["output"]["grid"]
    time_limit = 3.662
    safety = 0.45

    degree = 2
    candidates = [96, 128, 160]
    chosen = candidates[0]
    last_ok = None
    verification = {}

    for n in candidates:
        t_start = time.perf_counter()
        try:
            domain, V, uh, solver = _build_and_solve(comm, n=n, degree=degree, rtol=1e-10, ksp_type="cg", pc_type="hypre")
            elapsed = time.perf_counter() - t_start
            last_ok = (domain, V, uh, solver, n, elapsed)
            verification = _accuracy_verification(domain, uh, grid)
            if (time.perf_counter() - t0) + elapsed > (time_limit - safety):
                chosen = n
                break
            chosen = n
        except Exception:
            break

    if last_ok is None:
        domain, V, uh, solver = _build_and_solve(comm, n=40, degree=1, rtol=1e-9, ksp_type="preonly", pc_type="lu")
        chosen = 40
        verification = _accuracy_verification(domain, uh, grid)
    else:
        domain, V, uh, solver, _, _ = last_ok

    u_grid = _sample_on_grid(domain, uh, grid)

    ksp_type = solver.getType()
    pc_type = solver.getPC().getType()
    try:
        iterations = int(solver.getIterationNumber())
    except Exception:
        iterations = 1

    solver_info = {
        "mesh_resolution": int(chosen),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(1e-10),
        "iterations": int(iterations),
        "accuracy_verification": verification,
    }

    return {"u": u_grid, "solver_info": solver_info}
