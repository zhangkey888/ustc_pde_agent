import time
import math
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

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _sample_function(u_func: fem.Function, bbox, nx: int, ny: int) -> np.ndarray:
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([xx.ravel(), yy.ravel()])
    pts = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts[:, 0] = pts2[:, 0]
    pts[:, 1] = pts2[:, 1]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_idx = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_idx.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_idx, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = COMM.gather(values, root=0)
    if COMM.rank == 0:
        merged = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            merged = np.nan_to_num(merged)
        out = merged.reshape(ny, nx)
    else:
        out = None
    return COMM.bcast(out, root=0)


def _solve_once(mesh_resolution: int, degree: int, kappa_value: float = 2.0):
    msh = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    pi = math.pi

    u_exact_ufl = ufl.exp(x[0]) * ufl.cos(2.0 * pi * x[1])
    kappa = fem.Constant(msh, ScalarType(kappa_value))
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u_D = fem.Function(V)
    u_D.interpolate(lambda X: np.exp(X[0]) * np.cos(2.0 * np.pi * X[1]))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
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

    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=4000)

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("cg failed")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(msh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: np.exp(X[0]) * np.cos(2.0 * np.pi * X[1]))
    err_L2 = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    norm_L2 = fem.assemble_scalar(fem.form((u_ex) ** 2 * ufl.dx))
    err_L2 = math.sqrt(COMM.allreduce(err_L2, op=MPI.SUM))
    norm_L2 = math.sqrt(COMM.allreduce(norm_L2, op=MPI.SUM))
    rel_L2 = err_L2 / norm_L2 if norm_L2 > 0 else err_L2

    info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(solver.getTolerances()[0]),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(err_L2),
        "relative_l2_error": float(rel_L2),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    candidates = [(16, 2), (24, 2), (32, 2)]
    best_u = None
    best_info = None
    time_budget = 1.119
    margin = 0.15

    for mesh_res, degree in candidates:
        t0 = time.perf_counter()
        uh, info = _solve_once(mesh_res, degree, 2.0)
        best_u, best_info = uh, info
        elapsed = time.perf_counter() - start
        candidate_cost = time.perf_counter() - t0
        if info["l2_error"] <= 2.45e-03 and elapsed >= time_budget - margin:
            break
        if elapsed + max(candidate_cost, 0.05) > time_budget - margin:
            break

    u_grid = _sample_function(best_u, bbox, nx, ny)
    solver_info = {
        "mesh_resolution": best_info["mesh_resolution"],
        "element_degree": best_info["element_degree"],
        "ksp_type": best_info["ksp_type"],
        "pc_type": best_info["pc_type"],
        "rtol": best_info["rtol"],
        "iterations": best_info["iterations"],
    }
    return {"u": u_grid, "solver_info": solver_info}
