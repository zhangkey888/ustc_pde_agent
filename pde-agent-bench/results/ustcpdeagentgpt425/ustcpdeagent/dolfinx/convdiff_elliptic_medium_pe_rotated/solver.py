import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_numpy(x, y):
    return np.sin(2 * np.pi * (x + y)) * np.sin(np.pi * (x - y))


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    pts_local = []
    cells_local = []
    map_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            map_local.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if len(pts_local) > 0:
        vals = u_func.eval(np.asarray(pts_local, dtype=np.float64), np.asarray(cells_local, dtype=np.int32))
        values[np.asarray(map_local, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _sample_on_grid(u_func, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def _build_and_solve(n, degree, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    eps = ScalarType(0.05)
    beta_vec = np.array([3.0, 1.0], dtype=np.float64)
    beta = fem.Constant(msh, np.asarray(beta_vec, dtype=ScalarType))
    beta_norm = float(np.linalg.norm(beta_vec))

    u_exact_ufl = ufl.sin(2 * ufl.pi * (x[0] + x[1])) * ufl.sin(ufl.pi * (x[0] - x[1]))
    f_ufl = -eps * ufl.div(ufl.grad(u_exact_ufl)) + ufl.dot(beta, ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    tau = h / (2.0 * beta_norm)
    r_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx
    L = f_ufl * v * ufl.dx

    a += tau * ufl.dot(beta, ufl.grad(v)) * r_u * ufl.dx
    L += tau * ufl.dot(beta, ufl.grad(v)) * f_ufl * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(2 * np.pi * (X[0] + X[1])) * np.sin(np.pi * (X[0] - X[1])))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1)
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    eform = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(eform)
    l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "l2_error": float(l2_error),
    }
    return uh, solver_info


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    grid = case_spec["output"]["grid"]
    budget = 3.0
    if "time_limit" in case_spec:
        try:
            budget = float(case_spec["time_limit"])
        except Exception:
            budget = 3.0
    budget = min(budget, 3.2)

    candidates = [(48, 1), (64, 1), (80, 1), (64, 2)]
    chosen = None
    last = None

    for i, (n, degree) in enumerate(candidates):
        now = time.perf_counter()
        if now - start > 0.85 * budget and last is not None:
            break
        uh, info = _build_and_solve(n, degree)
        elapsed = time.perf_counter() - start
        last = (uh, info, elapsed)
        chosen = last
        if info["l2_error"] <= 5.10e-03:
            remaining = budget - elapsed
            if remaining < 0.7:
                break

    if chosen is None:
        uh, info = _build_and_solve(48, 1)
        chosen = (uh, info, time.perf_counter() - start)

    uh, info, _ = chosen
    u_grid = _sample_on_grid(uh, grid)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": info["mesh_resolution"],
            "element_degree": info["element_degree"],
            "ksp_type": info["ksp_type"],
            "pc_type": info["pc_type"],
            "rtol": info["rtol"],
            "iterations": info["iterations"],
        },
    }
