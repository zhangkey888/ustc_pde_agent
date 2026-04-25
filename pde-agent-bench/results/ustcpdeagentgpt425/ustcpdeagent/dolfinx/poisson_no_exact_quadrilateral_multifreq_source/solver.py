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
# special_notes: manufactured_solution
# ```

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

def _choose_params(case_spec: dict):
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    n_out = max(nx_out, ny_out)
    degree = 2
    if n_out <= 96:
        mesh_resolution = 36
    elif n_out <= 160:
        mesh_resolution = 48
    else:
        mesh_resolution = 56
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    return mesh_resolution, degree, ksp_type, pc_type, rtol


def _source_expr(x):
    return (
        ufl.sin(6.0 * math.pi * x[0]) * ufl.sin(5.0 * math.pi * x[1])
        + 0.4 * ufl.sin(11.0 * math.pi * x[0]) * ufl.sin(9.0 * math.pi * x[1])
    )


def _exact_numpy(x, y):
    lam1 = (6.0 * math.pi) ** 2 + (5.0 * math.pi) ** 2
    lam2 = (11.0 * math.pi) ** 2 + (9.0 * math.pi) ** 2
    return (
        np.sin(6.0 * math.pi * x) * np.sin(5.0 * math.pi * y) / lam1
        + 0.4 * np.sin(11.0 * math.pi * x) * np.sin(9.0 * math.pi * y) / lam2
    )


def _build_solver(mesh_resolution, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.quadrilateral
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = _source_expr(x)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    uh = fem.Function(V)
    return domain, a_form, L_form, bc, b, solver, uh


def _solve(a_form, L_form, bc, b, solver, uh):
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solver did not converge")
    except Exception:
        fallback = PETSc.KSP().create(uh.function_space.mesh.comm)
        fallback.setOperators(solver.getOperators()[0])
        fallback.setType("preonly")
        fallback.getPC().setType("lu")
        fallback.solve(b, uh.x.petsc_vec)
        solver = fallback

    uh.x.scatter_forward()
    return solver


def _eval_on_points(u_func: fem.Function, points: np.ndarray) -> np.ndarray:
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_local = []
    cells_local = []
    idx_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_local.append(points[i])
            cells_local.append(links[0])
            idx_local.append(i)

    values_local = np.full(points.shape[0], np.nan, dtype=np.float64)
    if points_local:
        vals = u_func.eval(
            np.asarray(points_local, dtype=np.float64),
            np.asarray(cells_local, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_local), -1)[:, 0]
        values_local[np.asarray(idx_local, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(values_local)
    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        values[mask] = arr[mask]
    return np.nan_to_num(values, nan=0.0)


def _sample_to_grid(u_func: fem.Function, grid: dict) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _eval_on_points(u_func, points)
    return vals.reshape(ny, nx)


def _verify_accuracy(u_func: fem.Function) -> dict:
    n = 41
    xs = np.linspace(0.0, 1.0, n, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n, dtype=np.float64)])
    uh = _eval_on_points(u_func, points)
    ue = _exact_numpy(X.ravel(), Y.ravel())
    err = uh - ue
    return {
        "grid_l2_error": float(np.sqrt(np.mean(err**2))),
        "grid_linf_error": float(np.max(np.abs(err))),
    }


def solve(case_spec: dict) -> dict:
    mesh_resolution, degree, ksp_type, pc_type, rtol = _choose_params(case_spec)
    domain, a_form, L_form, bc, b, solver, uh = _build_solver(
        mesh_resolution, degree, ksp_type, pc_type, rtol
    )
    solver = _solve(a_form, L_form, bc, b, solver, uh)

    u_grid = _sample_to_grid(uh, case_spec["output"]["grid"])
    verification = _verify_accuracy(uh)

    try:
        iterations = int(solver.getIterationNumber())
    except Exception:
        iterations = 0

    try:
        used_ksp = solver.getType()
    except Exception:
        used_ksp = ksp_type

    try:
        used_pc = solver.getPC().getType()
    except Exception:
        used_pc = pc_type

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(used_ksp),
        "pc_type": str(used_pc),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification_grid_l2_error": verification["grid_l2_error"],
        "verification_grid_linf_error": verification["grid_linf_error"],
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64),
        "solver_info": solver_info,
    }
