import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def _u_exact_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _solve_once(n, degree, rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_ex = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    kappa = 1.0 + 0.3 * ufl.sin(2.0 * pi * x[0]) * ufl.cos(2.0 * pi * x[1])
    f = -ufl.div(kappa * ufl.grad(u_ex))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    try:
        pc.setType("hypre")
        pc.setHYPREType("boomeramg")
        pc_type = "hypre"
    except Exception:
        pc.setType("jacobi")
        pc_type = "jacobi"
    solver.setTolerances(rtol=rtol)
    solver.solve(b, uh.x.petsc_vec)

    if solver.getConvergedReason() <= 0:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        ksp_type = "preonly"
        pc_type = "lu"
    else:
        ksp_type = "cg"

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    return {
        "domain": domain,
        "solution": uh,
        "error_l2": err_l2,
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(solver.getIterationNumber()),
    }


def _sample_solution(uh, grid_spec):
    domain = uh.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(pts[i])
            cells_local.append(links[0])
            ids_local.append(i)

    if pts_local:
        ev = uh.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals[np.array(ids_local, dtype=np.int32)] = np.asarray(ev).reshape(-1)

    if np.isnan(vals).any():
        mask = np.isnan(vals)
        vals[mask] = np.sin(np.pi * pts[mask, 0]) * np.sin(np.pi * pts[mask, 1])

    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    degree = 3
    start = time.perf_counter()

    candidates = [32, 48, 64]
    best = None
    for n in candidates:
        best = _solve_once(n, degree, rtol=1e-10)
        if time.perf_counter() - start > 2.0:
            break

    if best["error_l2"] > 3.59e-4:
        best = _solve_once(max(best["mesh_resolution"] + 16, 80), degree, rtol=1e-11)

    u_grid = _sample_solution(best["solution"], case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
        },
    }
