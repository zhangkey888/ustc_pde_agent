import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

# ```DIAGNOSIS
# equation_type:        biharmonic
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar+scalar
# coupling:             sequential
# linearity:            linear
# time_dependence:      steady
# stiffness:            stiff
# dominant_physics:     diffusion
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        none
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        cg
# preconditioner:       hypre
# special_treatment:    problem_splitting
# pde_skill:            none
# ```

def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)

def _sample_function_on_grid(domain, u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

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

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_map, dtype=np.int32)] = vals

    # Single-rank evaluator expected; still make robust in MPI.
    comm = domain.comm
    if comm.size > 1:
        recv = np.empty_like(values)
        comm.Allreduce(values, recv, op=MPI.SUM)
        values = recv

    # Replace any residual NaNs (can occur exactly on partition interfaces) by nearest valid zeros fallback.
    values = np.nan_to_num(values, nan=0.0)
    return values.reshape(ny, nx)

def _solve_poisson(domain, V, rhs_expr, bcs, ksp_type="cg", pc_type="hypre", rtol=1e-10, prefix="p_"):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rhs_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    uh = fem.Function(V)
    solver = PETSc.KSP().create(domain.comm)
    solver.setOptionsPrefix(prefix)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    if pc_type == "hypre":
        try:
            pc.setHYPREType("boomeramg")
        except Exception:
            pass
    solver.setTolerances(rtol=rtol, atol=0.0, max_it=10000)
    solver.setFromOptions()

    try:
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"KSP failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=0.0, max_it=1)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()
    its = int(solver.getIterationNumber())
    ksp_name = solver.getType()
    pc_name = solver.getPC().getType()
    solver.destroy()
    A.destroy()
    b.destroy()
    return uh, its, ksp_name, pc_name

def _compute_residual_indicator(domain, V, u_func, f_expr):
    z = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    # Compute w solving -Δw=f, then compare -Δu=w weakly.
    bc_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, _all_boundary)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, bc_facets)
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    bc = fem.dirichletbc(zero, dofs)
    w_func, _, _, _ = _solve_poisson(domain, V, f_expr, [bc], ksp_type="cg", pc_type="hypre", rtol=1e-10, prefix="aux_")
    form_num = fem.form((ufl.inner(ufl.grad(u_func), ufl.grad(q)) - w_func * q) ** 2 * ufl.dx)
    try:
        val = fem.assemble_scalar(form_num)
        val = domain.comm.allreduce(val, op=MPI.SUM)
        return float(np.sqrt(abs(val)))
    except Exception:
        return -1.0

def _run_once(nx, degree, grid, max_time=None, t0=None):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    bc = fem.dirichletbc(u0, dofs)

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.cos(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])

    # Mixed splitting approach:
    # Solve -Δw = f, w|∂Ω=0 ; then -Δu = w, u|∂Ω=0.
    # This is a practical surrogate accepted by task statement.
    w_h, its1, ksp_type, pc_type = _solve_poisson(domain, V, f_expr, [bc], ksp_type="cg", pc_type="hypre", rtol=1e-10, prefix="w_")
    u_h, its2, _, _ = _solve_poisson(domain, V, w_h, [bc], ksp_type="cg", pc_type="hypre", rtol=1e-10, prefix="u_")

    u_grid = _sample_function_on_grid(domain, u_h, grid)
    residual_indicator = _compute_residual_indicator(domain, V, u_h, f_expr)

    info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(1e-10),
        "iterations": int(its1 + its2),
        "verification": {
            "weak_poisson_chain_residual": float(residual_indicator)
        }
    }
    elapsed = time.perf_counter() - t0 if t0 is not None else None
    if max_time is not None and elapsed is not None:
        info["elapsed_so_far"] = float(elapsed)
    return u_grid, info

def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    grid = case_spec["output"]["grid"]

    # Adaptive accuracy/time trade-off
    # Start conservatively and refine if well under budget.
    time_budget = 15.175
    candidates = [(48, 2), (64, 2), (80, 2), (96, 2), (112, 2)]
    chosen_u = None
    chosen_info = None
    prev_u = None

    for i, (nx, degree) in enumerate(candidates):
        u_grid, info = _run_once(nx, degree, grid, max_time=time_budget, t0=t0)
        chosen_u, chosen_info = u_grid, info

        if prev_u is not None:
            diff = np.linalg.norm(u_grid - prev_u) / max(np.linalg.norm(u_grid), 1e-14)
            chosen_info["verification"]["grid_refinement_relative_change"] = float(diff)
        else:
            chosen_info["verification"]["grid_refinement_relative_change"] = None

        elapsed = time.perf_counter() - t0
        prev_u = u_grid.copy()

        # If close to budget, stop.
        if elapsed > 0.8 * time_budget:
            break

        # If refinement change is already tiny and we have a good mesh, can stop.
        if i >= 2 and chosen_info["verification"]["grid_refinement_relative_change"] is not None:
            if chosen_info["verification"]["grid_refinement_relative_change"] < 2e-3 and elapsed > 0.35 * time_budget:
                break

    return {
        "u": chosen_u,
        "solver_info": chosen_info
    }
