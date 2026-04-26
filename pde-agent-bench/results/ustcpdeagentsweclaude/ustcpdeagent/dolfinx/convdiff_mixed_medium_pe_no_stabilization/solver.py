import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])


def _probe_function(u_func, pts):
    """
    Evaluate scalar fem.Function at points pts with shape (N, 3).
    Returns array shape (N,).
    """
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(pts.shape[0], np.nan, dtype=np.float64)
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
        vals = np.asarray(vals).reshape(-1)
        values[np.array(eval_map, dtype=np.int32)] = vals
    return values


def _sample_on_uniform_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _build_and_solve(mesh_resolution, degree, use_supg, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = _manufactured_u_expr(x)

    eps = 0.02
    beta_vec = np.array([6.0, 2.0], dtype=np.float64)
    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(eps))

    lap_u = ufl.div(ufl.grad(u_exact_ufl))
    conv_u = ufl.dot(beta, ufl.grad(u_exact_ufl))
    f_ufl = -eps_c * lap_u + conv_u

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_ufl * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
        tau = h / (2.0 * beta_norm)
        r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        r_f = f_ufl
        a += tau * r_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * r_f * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("gmres")
    pc = solver.getPC()
    pc.setType("ilu")
    solver.setTolerances(rtol=rtol, atol=1.0e-14, max_it=5000)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    t0 = time.perf_counter()
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    solve_time = time.perf_counter() - t0

    its = solver.getIterationNumber()

    l2_local = fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "uh": uh,
        "l2_error": float(l2_err),
        "iterations": int(its),
        "solve_time": float(solve_time),
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "use_supg": bool(use_supg),
    }


def solve(case_spec: dict) -> dict:
    """
    Solve steady convection-diffusion equation with manufactured solution.
    Returns {"u": u_grid, "solver_info": {...}}
    """
    comm = MPI.COMM_WORLD
    t_start = time.perf_counter()

    candidates = [
        (64, 2, True, 1e-10),
    ]

    best = None
    time_budget = 9.0

    for mesh_resolution, degree, use_supg, rtol in candidates:
        result = _build_and_solve(mesh_resolution, degree, use_supg, rtol)
        elapsed = time.perf_counter() - t_start
        if best is None or result["l2_error"] < best["l2_error"]:
            best = result
        if result["l2_error"] <= 2.01e-3 and elapsed < time_budget:
            best = result
        if elapsed > time_budget:
            break

    if best is None:
        raise RuntimeError("Failed to compute a solution.")

    u_grid = _sample_on_uniform_grid(best["uh"], case_spec["output"]["grid"])

    if comm.size > 1:
        # Gather only from rank 0 if ever run in parallel; benchmark is expected serial.
        u_grid = comm.bcast(u_grid, root=0)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "stabilization": "SUPG" if best["use_supg"] else "none",
        "l2_error_estimate": best["l2_error"],
        "wall_time_sec": time.perf_counter() - t_start,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
