import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _rhs_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    return ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1]) + 0.4 * ufl.sin(11 * pi * x[0]) * ufl.sin(9 * pi * x[1])


def _exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    return (
        ufl.sin(6 * pi * x[0]) * ufl.sin(5 * pi * x[1]) / (((6 * pi) ** 2) + ((5 * pi) ** 2))
        + 0.4 * ufl.sin(11 * pi * x[0]) * ufl.sin(9 * pi * x[1]) / (((11 * pi) ** 2) + ((9 * pi) ** 2))
    )


def _sample_function(u_fun, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        evals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals[np.array(ids, dtype=np.int32)] = np.asarray(evals, dtype=np.float64).reshape(-1)

    comm = msh.comm
    if comm.size > 1:
        gathered = comm.allgather(vals)
        merged = np.full_like(vals, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        vals = merged

    if np.isnan(vals).any():
        nan_ids = np.where(np.isnan(vals))[0]
        for idx in nan_ids:
            x = pts[idx, 0]
            y = pts[idx, 1]
            if np.isclose(x, xmin) or np.isclose(x, xmax) or np.isclose(y, ymin) or np.isclose(y, ymax):
                vals[idx] = 0.0

    if np.isnan(vals).any():
        raise RuntimeError("Failed to sample solution on requested grid.")

    return vals.reshape(ny, nx)


def _build_and_solve(nx, degree, rtol=1e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [nx, nx],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = _rhs_ufl(msh)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    try:
        pc.setHYPREType("boomeramg")
    except Exception:
        pass
    solver.setTolerances(rtol=rtol)

    uh = fem.Function(V)

    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    try:
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
    except Exception:
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

    exact = _exact_ufl(msh)
    err_sq_local = fem.assemble_scalar(fem.form((uh - exact) ** 2 * ufl.dx))
    ref_sq_local = fem.assemble_scalar(fem.form((exact) ** 2 * ufl.dx))
    err_sq = comm.allreduce(err_sq_local, op=MPI.SUM)
    ref_sq = comm.allreduce(ref_sq_local, op=MPI.SUM)
    rel_l2 = math.sqrt(err_sq / ref_sq) if ref_sq > 0 else math.sqrt(err_sq)

    return {
        "mesh": msh,
        "u": uh,
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(solver.getIterationNumber()),
        "verified_rel_l2_error": float(rel_l2),
    }


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()
    budget = 3.919

    candidates = [(48, 2), (64, 2), (80, 2)]
    best = None
    for nx, degree in candidates:
        res = _build_and_solve(nx, degree, rtol=1e-10)
        best = res
        if time.perf_counter() - start > 0.8 * budget:
            break

    u_grid = _sample_function(best["u"], best["mesh"], case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
    }

    return {"u": u_grid, "solver_info": solver_info}
