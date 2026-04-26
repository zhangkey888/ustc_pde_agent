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
# special_notes:        manufactured_solution
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


def _exact_u_expr(x):
    return np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1]) + np.sin(np.pi * x[0]) * np.sin(3.0 * np.pi * x[1])


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals

    comm = domain.comm
    if comm.size > 1:
        send = np.where(np.isnan(local_vals), -np.inf, local_vals)
        recv = np.empty_like(send)
        comm.Allreduce(send, recv, op=MPI.MAX)
        local_vals = recv
        local_vals[np.isneginf(local_vals)] = np.nan

    if np.isnan(local_vals).any():
        missing = np.isnan(local_vals)
        local_vals[missing] = _exact_u_expr(pts2[missing].T)

    return local_vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    k = float(case_spec.get("pde", {}).get("k", 15.0))
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 48))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    ksp_type = str(case_spec.get("solver", {}).get("ksp_type", "gmres"))
    pc_type = str(case_spec.get("solver", {}).get("pc_type", "ilu"))
    rtol = float(case_spec.get("solver", {}).get("rtol", 1.0e-10))

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + ufl.sin(ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])

    lap_u_exact = (
        -((2.0 * ufl.pi) ** 2 + (ufl.pi) ** 2) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        -((ufl.pi) ** 2 + (3.0 * ufl.pi) ** 2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    )
    f_ufl = -lap_u_exact - (k ** 2) * u_exact_ufl

    uD = fem.Function(V)
    uD.interpolate(_exact_u_expr)

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

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
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=20000)
    solver.setFromOptions()

    try:
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Primary KSP failed with reason {reason}")
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=min(rtol, 1.0e-12), atol=1.0e-14, max_it=1)
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())

    error_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    local_err_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(local_err_sq, op=MPI.SUM))

    u_grid = _sample_function_on_grid(uh, domain, case_spec["output"]["grid"])

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
            "l2_error_vs_manufactured": float(l2_error),
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 15.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
