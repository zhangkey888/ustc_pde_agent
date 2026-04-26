import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

"""
DIAGNOSIS
equation_type: convection_diffusion
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: stiff
dominant_physics: mixed
peclet_or_reynolds: high
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: none
"""

"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: supg
time_method: none
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: convection_diffusion
"""


def _build_and_solve(n, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    eps = fem.Constant(domain, ScalarType(0.05))
    beta = fem.Constant(domain, np.array([2.0, 1.0], dtype=np.float64))
    f_expr = ufl.exp(-250.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-14)
    Pe = beta_norm * h / (2.0 * eps)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-14)
    tau = h / (2.0 * beta_norm) * (cothPe - 1.0 / (Pe + 1.0e-14))

    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_expr * v * ufl.dx

    r_trial = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    v_supg = ufl.dot(beta, ufl.grad(v))
    a += tau * r_trial * v_supg * ufl.dx
    L += tau * f_expr * v_supg * ufl.dx

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
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-13, max_it=5000)

    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve failed")
        used_ksp = ksp_type
        used_pc = pc_type
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        used_ksp = "preonly"
        used_pc = "lu"

    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())

    w = fem.Function(V)
    ww = ufl.TestFunction(V)
    F_res = (eps * ufl.inner(ufl.grad(uh), ufl.grad(ww)) + ufl.dot(beta, ufl.grad(uh)) * ww - f_expr * ww) * ufl.dx
    residual_norm = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(((-eps * ufl.div(ufl.grad(uh)) + ufl.dot(beta, ufl.grad(uh)) - f_expr) ** 2) * ufl.dx)), op=MPI.SUM))

    info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": iterations,
        "verification": {"cell_residual_l2": float(residual_norm)},
    }
    return domain, uh, info


def _probe_function(u_func, points_array):
    domain = u_func.function_space.mesh
    points = np.asarray(points_array, dtype=np.float64)
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    local_vals = np.full(points.shape[0], -1.0e300, dtype=np.float64)
    pts_local, cells_local, ids = [], [], []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids.append(i)

    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(pts_local), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    global_vals = np.empty_like(local_vals)
    domain.comm.Allreduce(local_vals, global_vals, op=MPI.MAX)
    global_vals[global_vals < -1.0e250] = 0.0
    return global_vals


def _sample_to_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    output_grid = case_spec["output"]["grid"]
    target_time = 47.641

    candidates = [96, 128, 160, 192]
    prev_grid = None
    best = None

    for n in candidates:
        _, uh, info = _build_and_solve(n=n, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-10)
        u_grid = _sample_to_grid(uh, output_grid)
        elapsed = time.perf_counter() - t0

        if prev_grid is not None:
            rel_change = np.linalg.norm(u_grid - prev_grid) / max(np.linalg.norm(u_grid), 1e-14)
        else:
            rel_change = np.inf
        info["verification"]["grid_relative_change_vs_previous"] = None if not np.isfinite(rel_change) else float(rel_change)
        info["verification"]["wall_time_sec"] = float(elapsed)

        best = (u_grid, info)
        prev_grid = u_grid.copy()

        if np.isfinite(rel_change) and rel_change < 8e-4 and elapsed > 0.2 * target_time:
            break
        if elapsed > 0.85 * target_time:
            break

    u_grid, info = best
    return {"u": u_grid, "solver_info": info}
