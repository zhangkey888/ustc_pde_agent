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
# element_or_basis: Lagrange_P2
# stabilization: supg
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: convection_diffusion / reaction_diffusion / biharmonic
# ```


def _manufactured_exact_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(pts.shape[0], -1.0e300, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    global_values = np.empty_like(local_values)
    domain.comm.Allreduce(local_values, global_values, op=MPI.MAX)
    global_values[global_values < -1.0e200] = 0.0
    return global_values.reshape(ny, nx)


def _build_and_solve(mesh_resolution=96, degree=2, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)

    epsilon = 0.005
    beta_np = np.array([18.0, 6.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_np)
    beta_norm = float(np.linalg.norm(beta_np))
    h = ufl.CellDiameter(domain)

    u_exact_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_expr = -epsilon * ufl.div(ufl.grad(u_exact_expr)) + ufl.dot(beta, ufl.grad(u_exact_expr))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    tau = h / (2.0 * beta_norm)
    a_std = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_expr * v * ufl.dx

    a_supg = tau * (ufl.dot(beta, ufl.grad(u))) * (ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * f_expr * (ufl.dot(beta, ufl.grad(v))) * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_exact_numpy)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=5000)
    if ksp_type == "gmres":
        solver.setGMRESRestart(200)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    used_ksp = ksp_type
    used_pc = pc_type
    used_rtol = rtol
    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError("iterative solver failed")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=1)
        solver.solve(b, uh.x.petsc_vec)
        used_ksp = "preonly"
        used_pc = "lu"
        used_rtol = 1e-12

    uh.x.scatter_forward()

    u_exact_fn = fem.Function(V)
    u_exact_fn.interpolate(_manufactured_exact_numpy)
    err_fn = fem.Function(V)
    err_fn.x.array[:] = uh.x.array - u_exact_fn.x.array
    local_err2 = fem.assemble_scalar(fem.form(ufl.inner(err_fn, err_fn) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(local_err2, op=MPI.SUM))

    return {
        "domain": domain,
        "uh": uh,
        "l2_error": float(l2_error),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": used_ksp,
        "pc_type": used_pc,
        "rtol": float(used_rtol),
        "iterations": int(solver.getIterationNumber()),
    }


def solve(case_spec: dict) -> dict:
    time_budget = 7.114
    candidates = [
        (56, 2, "gmres", "ilu", 1e-8),
        (64, 2, "gmres", "ilu", 1e-8),
    ]

    overall_t0 = time.perf_counter()
    best = None
    for params in candidates:
        t0 = time.perf_counter()
        result = _build_and_solve(*params)
        elapsed = time.perf_counter() - t0
        best = result
        if (time.perf_counter() - overall_t0) > 0.5 * time_budget:
            break
        if elapsed > 0.5 * time_budget:
            break

    u_grid = _sample_function_on_grid(best["uh"], best["domain"], case_spec["output"]["grid"])
    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall_time = time.perf_counter() - t0

    grid = case_spec["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    u_exact_grid = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    l2_error_grid = np.sqrt(np.mean((out["u"] - u_exact_grid) ** 2))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {l2_error_grid:.12e}")
        print(f"WALL_TIME: {wall_time:.12e}")
        print(out["solver_info"])
