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
# special_notes:        none
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
# special_treatment:    none
# pde_skill:            helmholtz
# ```

ScalarType = PETSc.ScalarType


def _rhs_expression(x):
    return 12.0 * (
        np.exp(-90.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        - np.exp(-90.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2))
    )


def _build_and_solve(n, degree=2, k=18.0, rtol=1.0e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = 12.0 * (
        ufl.exp(-90.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        - ufl.exp(-90.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2))
    )

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k**2) * u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("ilu")
    ksp.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
    ksp.setFromOptions()

    try:
        ksp.solve(b, uh.x.petsc_vec)
        reason = ksp.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"GMRES failed with reason {reason}")
        uh.x.scatter_forward()
        return domain, V, uh, {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": float(rtol),
            "iterations": int(ksp.getIterationNumber()),
        }
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        ksp.setTolerances(rtol=rtol)
        ksp.setFromOptions()
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        return domain, V, uh, {
            "mesh_resolution": int(n),
            "element_degree": int(degree),
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": float(rtol),
            "iterations": int(ksp.getIterationNumber()),
        }


def _sample_function(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = np.real(np.asarray(vals).reshape(-1))
        values_local[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    values_global = np.empty_like(values_local)
    comm.Allreduce(values_local, values_global, op=MPI.MAX)

    if np.isnan(values_global).any():
        mask = np.isnan(values_global)
        values_global[mask] = 0.0

    return values_global.reshape((ny, nx))


def _estimate_discrete_error(coarse_n, fine_n, degree=2, k=18.0):
    _, _, uh_c, _ = _build_and_solve(coarse_n, degree=degree, k=k, rtol=1.0e-9)
    domain_f, _, uh_f, _ = _build_and_solve(fine_n, degree=degree, k=k, rtol=1.0e-9)

    grid = {"nx": 81, "ny": 81, "bbox": [0.0, 1.0, 0.0, 1.0]}
    uc = _sample_function(mesh.create_unit_square(MPI.COMM_WORLD, coarse_n, coarse_n, cell_type=mesh.CellType.triangle), uh_c, grid) if False else None
    uf = _sample_function(domain_f, uh_f, grid)

    # Re-solve coarse on its own domain and sample there for consistency
    domain_c, _, uh_c2, _ = _build_and_solve(coarse_n, degree=degree, k=k, rtol=1.0e-9)
    uc = _sample_function(domain_c, uh_c2, grid)

    diff = uf - uc
    rel = np.linalg.norm(diff.ravel()) / max(np.linalg.norm(uf.ravel()), 1.0e-14)
    return float(rel)


def solve(case_spec: dict) -> dict:
    k = float(case_spec.get("pde", {}).get("k", 18.0))
    grid_spec = case_spec["output"]["grid"]

    t0 = time.perf_counter()

    # Start with a robust/high-accuracy default and adapt upward if very fast
    candidate_meshes = [80, 112, 144]
    chosen_n = candidate_meshes[0]
    degree = 2
    solver_info = None
    domain = V = uh = None

    for n in candidate_meshes:
        ts = time.perf_counter()
        domain_i, V_i, uh_i, info_i = _build_and_solve(n, degree=degree, k=k, rtol=1.0e-9)
        elapsed = time.perf_counter() - ts
        domain, V, uh, solver_info = domain_i, V_i, uh_i, info_i
        chosen_n = n
        # If solve is already moderately expensive, keep this resolution; otherwise use more accuracy.
        if elapsed > 8.0:
            break

    # Mandatory accuracy verification: mesh-consistency check on two nested resolutions if affordable.
    verification = {}
    total_elapsed = time.perf_counter() - t0
    if total_elapsed < 120.0:
        try:
            coarse_n = max(32, chosen_n // 2)
            verification["mesh_consistency_rel_error"] = _estimate_discrete_error(coarse_n, chosen_n, degree=degree, k=k)
            verification["verification_type"] = "coarse_fine_grid_comparison"
        except Exception as exc:
            verification["verification_type"] = "mesh_consistency_failed"
            verification["verification_message"] = str(exc)

    u_grid = _sample_function(domain, uh, grid_spec)

    solver_info = dict(solver_info)
    solver_info.update(verification)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 18.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
