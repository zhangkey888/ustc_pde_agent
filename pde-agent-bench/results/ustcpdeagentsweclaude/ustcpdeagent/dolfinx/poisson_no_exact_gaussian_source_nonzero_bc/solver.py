import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _g(x):
    return np.sin(2.0 * np.pi * x[0]) + 0.5 * np.cos(2.0 * np.pi * x[1])


def _eval_on_points(domain, uh, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells = []
    ids = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells.append(links[0])
            ids.append(i)

    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    if points_on_proc:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64),
                       np.asarray(cells, dtype=np.int32))
        values[np.asarray(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)
    return values


def _sample_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    vals = _eval_on_points(domain, uh, points)
    if np.isnan(vals).any():
        mask = np.isnan(vals)
        vals[mask] = np.sin(2.0 * np.pi * points[mask, 0]) + 0.5 * np.cos(2.0 * np.pi * points[mask, 1])
    return vals.reshape(ny, nx)


def _build_and_solve(mesh_resolution, degree, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.exp(-180.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))

    a_ufl = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ufl = f_expr * v * ufl.dx
    a = fem.form(a_ufl)
    L = fem.form(L_ufl)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_g)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("CG failed")
        ksp_type = str(ksp.getType())
        pc_type = str(ksp.getPC().getType())
        iterations = int(ksp.getIterationNumber())
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)
        ksp_type = str(ksp.getType())
        pc_type = str(ksp.getPC().getType())
        iterations = 1

    uh.x.scatter_forward()

    r = A.createVecRight()
    A.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    residual_norm = float(r.norm())

    s = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    bpts = np.vstack((
        np.column_stack((s, np.zeros_like(s), np.zeros_like(s))),
        np.column_stack((s, np.ones_like(s), np.zeros_like(s))),
        np.column_stack((np.zeros_like(s), s, np.zeros_like(s))),
        np.column_stack((np.ones_like(s), s, np.zeros_like(s))),
    ))
    ub = _eval_on_points(domain, uh, bpts)
    gb = np.sin(2.0 * np.pi * bpts[:, 0]) + 0.5 * np.cos(2.0 * np.pi * bpts[:, 1])
    boundary_max_error = float(np.nanmax(np.abs(ub - gb)))

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": iterations,
        "verification": {
            "residual_norm": residual_norm,
            "boundary_max_error": boundary_max_error,
        },
    }
    return domain, uh, solver_info


def solve(case_spec: dict) -> dict:
    start = time.time()
    candidates = [(80, 2), (96, 2), (112, 2)]
    best = None

    for mesh_resolution, degree in candidates:
        best = _build_and_solve(mesh_resolution, degree, 1e-10)
        if time.time() - start > 5.2:
            break

    domain, uh, solver_info = best
    u_grid = _sample_grid(domain, uh, case_spec["output"]["grid"])
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
