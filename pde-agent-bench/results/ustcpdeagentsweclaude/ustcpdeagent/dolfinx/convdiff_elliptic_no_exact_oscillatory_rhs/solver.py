import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: non_stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: none
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P1
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion
# ```


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        merged = np.full((pts2.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        # Fill any unresolved boundary points with zero Dirichlet value
        merged[np.isnan(merged)] = 0.0
        return merged.reshape(ny, nx)
    return None


def _solve_once(comm, n, degree=1, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(0.05)
    beta_vec = np.array([3.0, 3.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_vec)
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(5.0 * ufl.pi * x[1])

    tdim = domain.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), bdofs, V)

    h = ufl.CellDiameter(domain)
    beta_norm = float(np.linalg.norm(beta_vec))
    tau = h / (2.0 * beta_norm)

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
        + tau * ufl.inner(beta, ufl.grad(u)) * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_expr * v * ufl.dx
        + tau * f_expr * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=2000)
    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    reason = solver.getConvergedReason()
    iterations = solver.getIterationNumber()

    if reason <= 0:
        solver.destroy()
        A.destroy()
        b.destroy()

        solver = PETSc.KSP().create(comm)
        solver.setOperators(A := petsc.assemble_matrix(a_form, bcs=[bc]))
        A.assemble()
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12)
        with b := petsc.create_vector(L_form.function_spaces):
            pass

    if reason <= 0:
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
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        iterations = solver.getIterationNumber()
        ksp_type = "preonly"
        pc_type = "lu"

    # Residual-based verification
    r = b.duplicate()
    A.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    res_norm = r.norm()
    b_norm = b.norm()
    rel_res = res_norm / max(b_norm, 1e-16)

    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "iterations": int(iterations),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "solve_time": float(solve_time),
        "verification": {
            "linear_relative_residual": float(rel_res),
            "ksp_converged_reason": int(reason),
        },
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    grid_spec = case_spec["output"]["grid"]

    # Adaptive accuracy-time trade-off:
    # start reasonably fine for high-Pe SUPG case; refine if budget seems generous.
    candidate_meshes = [96, 128, 160]
    degree = 1
    time_limit = 205.064

    best = None
    prev_grid = None
    verification = {}

    t_start = time.perf_counter()

    for n in candidate_meshes:
        result = _solve_once(comm, n=n, degree=degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9)
        u_grid = _sample_function_on_grid(result["domain"], result["uh"], grid_spec)

        if comm.rank == 0 and prev_grid is not None:
            diff = np.linalg.norm(u_grid - prev_grid) / max(np.linalg.norm(u_grid), 1e-14)
            verification[f"grid_change_n{n}"] = float(diff)
        prev_grid = u_grid if comm.rank == 0 else None
        best = (n, result, u_grid)

        elapsed = time.perf_counter() - t_start
        # Use more accuracy if there is ample time and changes are not yet very small
        if elapsed > 0.45 * time_limit:
            break

    n, result, u_grid = best

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(result["rtol"]),
        "iterations": int(result["iterations"]),
    }
    solver_info.update(result["verification"])
    solver_info.update(verification)

    if comm.rank == 0:
        return {"u": np.asarray(u_grid, dtype=np.float64), "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}
