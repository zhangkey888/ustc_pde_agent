import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _manufactured_u(x):
    return np.exp(6.0 * x[0]) * np.sin(np.pi * x[1])


def _sample_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64),
                       np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        values[np.array(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64)

    comm = domain.comm
    gathered = comm.allgather(values)
    final = np.full_like(values, np.nan)
    for arr in gathered:
        mask = ~np.isnan(arr)
        final[mask] = arr[mask]

    return final.reshape((ny, nx))


def _solve_once(n, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = -(ufl.div(ufl.grad(u_exact_ufl)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = fem.form(ufl.inner(f_ufl, v) * ufl.dx)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_u)
    bc = fem.dirichletbc(u_bc, bdofs)

    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
        solver.solve(b, uh.x.petsc_vec)
        reason = solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"Iterative solver failed with reason {reason}")
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol)
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"

    uh.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(
        fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    )
    exact_L2_local = fem.assemble_scalar(
        fem.form((u_exact_ufl) ** 2 * ufl.dx)
    )
    err_L2 = np.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    exact_L2 = np.sqrt(comm.allreduce(exact_L2_local, op=MPI.SUM))
    rel_L2 = err_L2 / exact_L2 if exact_L2 > 0 else err_L2

    iterations = solver.getIterationNumber()

    return {
        "domain": domain,
        "uh": uh,
        "abs_L2_error": float(err_L2),
        "rel_L2_error": float(rel_L2),
        "iterations": int(iterations),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    target_time = 0.82
    candidates = [(24, 1), (32, 1), (40, 1), (24, 2), (32, 2), (40, 2)]
    chosen = None
    elapsed = 0.0

    for n, degree in candidates:
        run = _solve_once(n=n, degree=degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)
        elapsed = time.perf_counter() - t0
        chosen = run
        if elapsed > target_time:
            break

    if chosen is None:
        chosen = _solve_once(n=32, degree=2, ksp_type="cg", pc_type="hypre", rtol=1.0e-10)

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(chosen["domain"], chosen["uh"], grid)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": chosen["mesh_resolution"],
            "element_degree": chosen["element_degree"],
            "ksp_type": chosen["ksp_type"],
            "pc_type": chosen["pc_type"],
            "rtol": chosen["rtol"],
            "iterations": chosen["iterations"],
            "abs_L2_error": chosen["abs_L2_error"],
            "rel_L2_error": chosen["rel_L2_error"],
        },
    }

    return result
