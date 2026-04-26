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
# solution_regularity: boundary_layer
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion
# ```

ScalarType = PETSc.ScalarType


def _evaluate_on_grid(u_func, domain, nx, ny, bbox):
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.zeros(nx * ny, dtype=np.float64)
    found = np.zeros(nx * ny, dtype=np.int32)
    points_on_proc, cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        idx = np.array(ids, dtype=np.int32)
        local_vals[idx] = vals
        found[idx] = 1

    global_vals = np.zeros_like(local_vals)
    global_found = np.zeros_like(found)
    domain.comm.Allreduce(local_vals, global_vals, op=MPI.SUM)
    domain.comm.Allreduce(found, global_found, op=MPI.SUM)
    global_vals[global_found == 0] = np.nan
    return global_vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    eps = 0.05
    beta_np = np.array([4.0, 0.0], dtype=np.float64)
    beta_norm = float(np.linalg.norm(beta_np))

    # Chosen to balance accuracy and time budget for the medium-Pe manufactured case
    mesh_resolution = 80
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_exact = ufl.exp(2.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    lap_u_exact = (4.0 - ufl.pi**2) * u_exact
    f_expr = -eps * lap_u_exact + beta_np[0] * (2.0 * u_exact)

    beta = fem.Constant(domain, ScalarType(beta_np))
    eps_c = fem.Constant(domain, ScalarType(eps))

    h = ufl.CellDiameter(domain)
    tau = 1.0 / (2.0 * beta_norm / h + 4.0 * eps / (h * h) + 1.0e-14)

    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_expr * v * ufl.dx

    strong_res_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    test_stream = ufl.dot(beta, ufl.grad(v))

    a = a_std + tau * strong_res_u * test_stream * ufl.dx
    L = L_std + tau * f_expr * test_stream * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(2.0 * X[0]) * np.sin(np.pi * X[1]))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    uh = fem.Function(V)

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
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()
    iterations = int(solver.getIterationNumber())

    # Accuracy verification
    err_sq = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    exact_sq = fem.assemble_scalar(fem.form(u_exact ** 2 * ufl.dx))
    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    exact_sq = comm.allreduce(exact_sq, op=MPI.SUM)
    rel_l2 = float(np.sqrt(err_sq / (exact_sq + 1.0e-30)))

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    u_grid = _evaluate_on_grid(uh, domain, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "verification_rel_l2": rel_l2,
    }

    return {"u": u_grid, "solver_info": solver_info}
