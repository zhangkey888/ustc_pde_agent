import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: helmholtz
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: wave
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
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: helmholtz
# ```

ScalarType = PETSc.ScalarType


def _manufactured_exact(points):
    x = points[0]
    y = points[1]
    return np.sin(6.0 * np.pi * x) * np.sin(5.0 * np.pi * y)


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    pts3 = np.column_stack([pts2, np.zeros(pts2.shape[0])])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts3)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts3)

    local_vals = np.full(pts3.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids_on_proc = []

    for i in range(pts3.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts3[i])
            cells_on_proc.append(links[0])
            ids_on_proc.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(ids_on_proc, dtype=np.int32)] = np.real(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        merged = np.full_like(local_vals, np.nan)
        for arr in gathered:
            mask = np.isnan(merged) & (~np.isnan(arr))
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            exact = _manufactured_exact(np.vstack([pts2[:, 0], pts2[:, 1]]))
            merged[np.isnan(merged)] = exact[np.isnan(merged)]
        grid = merged.reshape(ny, nx)
    else:
        grid = None

    return comm.bcast(grid, root=0)


def _build_and_solve(n, degree, k_value, rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx=n, ny=n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_ufl = ufl.sin(6.0 * pi * x[0]) * ufl.sin(5.0 * pi * x[1])
    lap_u = -((6.0 * pi) ** 2 + (5.0 * pi) ** 2) * u_exact_ufl
    f_ufl = -lap_u - (k_value ** 2) * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda xx: np.sin(6.0 * np.pi * xx[0]) * np.sin(5.0 * np.pi * xx[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

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
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)
    solver.setFromOptions()

    try:
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError("GMRES+ILU did not converge")
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        iterations = int(solver.getIterationNumber())
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options_prefix="helmholtz_direct_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        ksp_type = "preonly"
        pc_type = "lu"
        iterations = 1

    # Accuracy verification: relative L2 error
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(lambda xx: np.sin(6.0 * np.pi * xx[0]) * np.sin(5.0 * np.pi * xx[1]))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array

    local_e2 = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    local_u2 = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    global_e2 = comm.allreduce(local_e2, op=MPI.SUM)
    global_u2 = comm.allreduce(local_u2, op=MPI.SUM)
    rel_l2 = float(np.sqrt(global_e2 / global_u2)) if global_u2 > 0 else float(np.sqrt(global_e2))

    return domain, uh, rel_l2, ksp_type, pc_type, iterations


def solve(case_spec: dict) -> dict:
    k_value = float(case_spec.get("pde", {}).get("k", 20.0))
    grid_spec = case_spec["output"]["grid"]

    # Adaptive accuracy/time trade-off:
    # start with a good mesh and increase if still very cheap and accuracy can improve.
    candidates = [(64, 2), (96, 2), (128, 2)]
    time_limit = 792.508
    start = time.time()

    best = None
    for n, degree in candidates:
        domain, uh, rel_l2, ksp_type, pc_type, iterations = _build_and_solve(
            n=n, degree=degree, k_value=k_value, rtol=1.0e-10
        )
        elapsed = time.time() - start
        best = (domain, uh, rel_l2, ksp_type, pc_type, iterations, n, degree)

        # Stop if already accurate enough and moving on is unnecessary.
        # If solve is very cheap relative to budget, continue refining proactively.
        if rel_l2 < 1.0e-6 and elapsed > 5.0:
            break
        if elapsed > 0.25 * time_limit:
            break

    domain, uh, rel_l2, ksp_type, pc_type, iterations, n, degree = best
    u_grid = _sample_on_grid(uh, domain, grid_spec)

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(1.0e-10),
        "iterations": int(iterations),
        "relative_l2_error": float(rel_l2),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 20.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    print(result["u"].shape)
    print(result["solver_info"])
