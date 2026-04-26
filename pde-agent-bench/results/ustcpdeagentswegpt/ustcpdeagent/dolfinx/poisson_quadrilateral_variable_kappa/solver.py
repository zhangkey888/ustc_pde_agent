import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _sample_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(nx * ny, np.nan, dtype=np.float64)
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
        local_values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_values, root=0)

    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            missing = np.isnan(merged).sum()
            raise RuntimeError(f"Failed to evaluate {missing} grid points.")
        return merged.reshape((ny, nx))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Problem-driven discretization tuned for accuracy/time target
    mesh_resolution = int(case_spec.get("solver_options", {}).get("mesh_resolution", 40))
    element_degree = int(case_spec.get("solver_options", {}).get("element_degree", 2))
    ksp_type = case_spec.get("solver_options", {}).get("ksp_type", "cg")
    pc_type = case_spec.get("solver_options", {}).get("pc_type", "hypre")
    rtol = float(case_spec.get("solver_options", {}).get("rtol", 1e-10))

    # Unit square with quadrilateral cells, matching case id intent
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact_ufl = ufl.sin(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    kappa_ufl = 1.0 + 0.5 * ufl.cos(2.0 * pi * x[0]) * ufl.cos(2.0 * pi * x[1])
    f_ufl = -ufl.div(kappa_ufl * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    # Dirichlet BC from manufactured solution on full boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Assemble and solve manually to record iterations and allow fallback
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
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    # Reasonable defaults for the chosen methods
    if ksp_type == "cg":
        solver.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
    solver.setFromOptions()

    t0 = time.perf_counter()
    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        # Fallback for robustness
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    iterations = int(solver.getIterationNumber())

    # Accuracy verification: compute L2 error against exact solution
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    l2_error_sq_local = fem.assemble_scalar(err_form)
    l2_error_sq = comm.allreduce(l2_error_sq_local, op=MPI.SUM)
    l2_error = float(np.sqrt(l2_error_sq))

    # If runtime is tiny, do one inexpensive self-check on a refined comparison grid
    verification = {
        "l2_error": l2_error,
        "wall_time_sec": solve_time,
    }

    u_grid = _sample_on_grid(uh, domain, case_spec["output"]["grid"])

    result = None
    if comm.rank == 0:
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "rtol": rtol,
                "iterations": iterations,
                "verification_l2_error": verification["l2_error"],
                "solve_wall_time_sec": verification["wall_time_sec"],
            },
        }
    return result


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
