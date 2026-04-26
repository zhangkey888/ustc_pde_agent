import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _boundary_values(x):
    return np.sin(2.0 * np.pi * x[0]) + 0.5 * np.cos(2.0 * np.pi * x[1])


def _probe_function(u_func, points):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points.T)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    npts = points.shape[1]
    vals = np.full(npts, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(npts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        arr = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals[np.array(eval_map, dtype=np.int32)] = np.asarray(arr).reshape(-1).real
    return vals


def _sample_on_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.vstack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)

    if np.isnan(vals).any():
        eps = 1e-12
        pts[0] = np.clip(pts[0], xmin + eps, xmax - eps)
        pts[1] = np.clip(pts[1], ymin + eps, ymax - eps)
        vals = _probe_function(u_func, pts)

    vals = np.nan_to_num(vals, nan=0.0)
    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    solver_cfg = case_spec.get("solver", {})
    mesh_resolution = int(solver_cfg.get("mesh_resolution", 96))
    element_degree = int(solver_cfg.get("element_degree", 2))
    rtol = float(solver_cfg.get("rtol", 1e-10))

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.exp(-180.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))

    g = fem.Function(V)
    g.interpolate(_boundary_values)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(g, dofs)

    a_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = fem.form(ufl.inner(f_expr, v) * ufl.dx)

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

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=5000)

    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("Iterative solve did not converge")
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
        iterations = ksp.getIterationNumber()
    except Exception:
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(b, uh.x.petsc_vec)
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
        iterations = ksp.getIterationNumber()

    uh.x.scatter_forward()

    residual = petsc.create_vector(L_form.function_spaces)
    A.mult(uh.x.petsc_vec, residual)
    residual.axpy(-1.0, b)
    residual_norm = residual.norm()

    ncheck = 128
    s = np.linspace(0.0, 1.0, ncheck)
    bpts = np.concatenate(
        [
            np.vstack([s, np.zeros_like(s), np.zeros_like(s)]).T,
            np.vstack([s, np.ones_like(s), np.zeros_like(s)]).T,
            np.vstack([np.zeros_like(s), s, np.zeros_like(s)]).T,
            np.vstack([np.ones_like(s), s, np.zeros_like(s)]).T,
        ],
        axis=0,
    )
    boundary_num = _probe_function(uh, bpts.T)
    boundary_ex = _boundary_values(bpts.T)
    boundary_max_error = float(np.nanmax(np.abs(boundary_num - boundary_ex)))

    u_grid = _sample_on_grid(uh, case_spec["output"]["grid"])

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": rtol,
            "iterations": int(iterations),
            "verification": {
                "linear_residual_norm": float(residual_norm),
                "boundary_max_error": boundary_max_error,
            },
        },
    }
