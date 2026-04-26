import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type:        convection_diffusion
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            non_stiff
# dominant_physics:     mixed
# peclet_or_reynolds:   moderate
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        manufactured_solution
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_Q2
# stabilization:        supg
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    none
# pde_skill:            convection_diffusion / reaction_diffusion
# ```


ScalarType = PETSc.ScalarType


def _manufactured_exprs(msh, eps_value=0.25, beta=(1.0, 0.5)):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    beta_ufl = ufl.as_vector((ScalarType(beta[0]), ScalarType(beta[1])))
    f_expr = -eps_value * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta_ufl, ufl.grad(u_exact))
    return u_exact, f_expr, beta_ufl


def _build_and_solve(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.quadrilateral)

    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    eps_value = 0.25
    beta_tuple = (1.0, 0.5)
    u_exact_ufl, f_ufl, beta_ufl = _manufactured_exprs(msh, eps_value=eps_value, beta=beta_tuple)

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_h = beta_norm * h / (2.0 * eps_value)

    # Smooth SUPG activation; negligible for low/moderate Pe, present if needed
    tau = h / (2.0 * beta_norm + 1.0e-12) * ufl.max_value(0.0, 1.0 - 1.0 / ufl.max_value(Pe_h, 1.0))

    strong_res_u = -eps_value * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))
    strong_res_f = f_ufl

    a = (
        eps_value * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
        + tau * strong_res_u * ufl.dot(beta_ufl, ufl.grad(v)) * ufl.dx
    )
    L = (
        f_ufl * v * ufl.dx
        + tau * strong_res_f * ufl.dot(beta_ufl, ufl.grad(v)) * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=1000)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Accuracy verification: L2 error against exact manufactured solution
    err_L2_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    ref_L2_form = fem.form((u_exact_ufl) ** 2 * ufl.dx)
    err_sq = fem.assemble_scalar(err_L2_form)
    ref_sq = fem.assemble_scalar(ref_L2_form)
    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    ref_sq = comm.allreduce(ref_sq, op=MPI.SUM)
    l2_error = float(np.sqrt(err_sq))
    rel_l2_error = float(np.sqrt(err_sq / ref_sq)) if ref_sq > 0 else 0.0

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
        "iterations": int(solver.getIterationNumber()),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "mesh_resolution": n,
        "element_degree": degree,
    }


def _probe_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
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
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    # Gather across ranks and combine
    comm = msh.comm
    gathered = comm.allgather(values)
    combined = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        combined[mask] = arr[mask]

    # For exact boundary points missed due to geometric tolerance, fill analytically
    missing = np.isnan(combined)
    if np.any(missing):
        x = pts[missing, 0]
        y = pts[missing, 1]
        combined[missing] = np.sin(np.pi * x) * np.sin(np.pi * y)

    return combined.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    # Adaptive accuracy/time trade-off:
    # start sufficiently accurate, refine if expected error is still above target.
    candidate_resolutions = [40, 56, 72, 88]
    best = None
    target = 6.26e-4 * 0.6  # safety margin

    for n in candidate_resolutions:
        result = _build_and_solve(n)
        best = result
        if result["l2_error"] <= target:
            break

    u_grid = _probe_function(best["uh"], bbox, nx, ny)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["iterations"]),
        "l2_error_verification": float(best["l2_error"]),
        "relative_l2_error_verification": float(best["rel_l2_error"]),
    }

    return {"u": u_grid, "solver_info": solver_info}
