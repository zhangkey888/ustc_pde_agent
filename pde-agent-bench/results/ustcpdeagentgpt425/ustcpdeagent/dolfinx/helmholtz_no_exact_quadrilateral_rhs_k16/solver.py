import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type:        helmholtz
# spatial_dim:          2
# domain_geometry:      rectangle
# unknowns:             scalar
# coupling:             none
# linearity:            linear
# time_dependence:      steady
# stiffness:            N/A
# dominant_physics:     wave
# peclet_or_reynolds:   N/A
# solution_regularity:  smooth
# bc_type:              all_dirichlet
# special_notes:        variable_coeff
# ```
#
# ```METHOD
# spatial_method:       fem
# element_or_basis:     Lagrange_P2
# stabilization:        none
# time_method:          none
# nonlinear_solver:     none
# linear_solver:        gmres
# preconditioner:       ilu
# special_treatment:    problem_splitting
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType
COMM = MPI.COMM_WORLD


def _rhs_callable(x):
    return np.sin(6.0 * np.pi * x[0]) * np.cos(5.0 * np.pi * x[1])


def _build_and_solve(n, degree=2, k=16.0, prefer_direct=False):
    domain = mesh.create_rectangle(
        COMM,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.cos(5.0 * ufl.pi * x[1])

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k**2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as b_loc:
        b_loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setTolerances(rtol=1e-9, atol=1e-12, max_it=5000)

    if prefer_direct:
        solver.setType("preonly")
        solver.getPC().setType("lu")
        ksp_type = "preonly"
        pc_type = "lu"
    else:
        solver.setType("gmres")
        solver.getPC().setType("ilu")
        ksp_type = "gmres"
        pc_type = "ilu"

    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if (reason is None) or (reason <= 0):
            raise RuntimeError(f"KSP did not converge, reason={reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    its = int(solver.getIterationNumber())

    return {
        "domain": domain,
        "V": V,
        "u": uh,
        "iterations": its,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(solver.getTolerances()[0]),
    }


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.real(vals.reshape(-1))

    gathered = COMM.gather(local_vals, root=0)
    if COMM.rank == 0:
        merged = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(merged) & (~np.isnan(arr))
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution on some output grid points.")
        grid = merged.reshape(ny, nx)
    else:
        grid = None

    return COMM.bcast(grid, root=0)


def _compute_refinement_error(coarse, fine):
    Vc = coarse["V"]
    uf = fine["u"]
    uc = coarse["u"]

    u_interp = fem.Function(Vc)
    u_interp.interpolate(uf)
    diff = fem.Function(Vc)
    diff.x.array[:] = uc.x.array - u_interp.x.array
    diff.x.scatter_forward()

    err_form = fem.form(ufl.inner(diff, diff) * ufl.dx)
    ref_form = fem.form(ufl.inner(u_interp, u_interp) * ufl.dx)
    err = np.sqrt(COMM.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref = np.sqrt(COMM.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    return float(err / (ref + 1e-14))


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    pde = case_spec.get("pde", {})
    k = float(pde.get("wavenumber", case_spec.get("wavenumber", 16.0)))
    if not np.isfinite(k):
        k = 16.0

    degree = 2
    start = time.perf_counter()

    base_n = 40
    verify_n = 52

    chosen = _build_and_solve(base_n, degree=degree, k=k, prefer_direct=False)
    chosen_n = base_n
    try:
        fine = _build_and_solve(verify_n, degree=degree, k=k, prefer_direct=False)
        chosen_err = _compute_refinement_error(chosen, fine)
        if (time.perf_counter() - start) < 200.0:
            chosen = fine
            chosen_n = verify_n
    except Exception:
        chosen_err = np.nan

    u_grid = _sample_on_grid(chosen["domain"], chosen["u"], output_grid)

    solver_info = {
        "mesh_resolution": int(chosen_n),
        "element_degree": int(degree),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "accuracy_check": {
            "type": "mesh_refinement_relative_L2",
            "estimated_relative_error": None if chosen_err is None or np.isnan(chosen_err) else float(chosen_err),
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"wavenumber": 16.0, "time": None},
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if COMM.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
