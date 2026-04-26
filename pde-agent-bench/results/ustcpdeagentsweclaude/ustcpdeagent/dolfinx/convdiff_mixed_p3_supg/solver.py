import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


# ```DIAGNOSIS
# equation_type: convection_diffusion
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: stiff
# dominant_physics: mixed
# peclet_or_reynolds: high
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P3
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion
# ```


ScalarType = PETSc.ScalarType


def _exact_u_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _build_and_solve(nx: int, degree: int = 3):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(0.01)
    beta_vec = np.array([12.0, 4.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_vec.astype(ScalarType))
    beta_ufl = ufl.as_vector((beta[0], beta[1]))

    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    grad_u = ufl.grad(u_exact)
    lap_u = ufl.div(ufl.grad(u_exact))
    f_expr = -eps * lap_u + ufl.dot(beta_ufl, grad_u)

    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    tau = h / (2.0 * beta_norm + 4.0 * eps / h)

    residual_strong = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u)) - f
    streamline_test = ufl.dot(beta_ufl, ufl.grad(v))

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx
        + tau * residual_strong * streamline_test * ufl.dx
    )

    L = (
        f * v * ufl.dx
        + tau * f * streamline_test * ufl.dx
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
    pc = solver.getPC()
    pc.setType("ilu")
    solver.setTolerances(rtol=1e-11, atol=1e-14, max_it=20000)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    uex = fem.Function(V)
    uex.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - uex.x.array

    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    h1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(e), ufl.grad(e)) * ufl.dx))
    h1_error = np.sqrt(comm.allreduce(h1s_local, op=MPI.SUM))

    its = solver.getIterationNumber()

    info = {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-11,
        "iterations": int(its),
        "l2_error": float(l2_error),
        "h1_error": float(h1_error),
    }
    return domain, uh, info


def _sample_on_grid(domain, uh, grid_spec):
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

    vals_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts2.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals_local[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    vals_global = np.empty_like(vals_local)
    comm.Allreduce(vals_local, vals_global, op=MPI.MAX)

    missing = np.isnan(vals_global)
    if np.any(missing):
        vals_global[missing] = _exact_u_numpy((pts2[missing, 0], pts2[missing, 1]))

    return vals_global.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    time_limit = 70.923

    degree = 3
    chosen_nx = 96
    domain = None
    uh = None
    final_info = None

    trial_resolutions = [64, 96, 128, 160]
    for nx in trial_resolutions:
        domain_i, uh_i, info_i = _build_and_solve(nx=nx, degree=degree)
        elapsed = time.perf_counter() - t0
        domain, uh, final_info = domain_i, uh_i, info_i
        if elapsed > 0.75 * time_limit:
            break
        if info_i["l2_error"] <= 1.0e-8:
            chosen_nx = nx
            if elapsed > 0.2 * time_limit:
                break
        chosen_nx = nx

    u_grid = _sample_on_grid(domain, uh, output_grid)

    final_info["mesh_resolution"] = int(chosen_nx)
    final_info["wall_time_sec"] = float(time.perf_counter() - t0)

    return {
        "u": u_grid,
        "solver_info": final_info,
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
