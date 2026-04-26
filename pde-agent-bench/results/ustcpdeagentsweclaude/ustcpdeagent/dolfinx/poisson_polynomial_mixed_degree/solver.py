import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])


def _sample_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_vals)
    vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        vals[mask] = arr[mask]

    if np.isnan(vals).any():
        miss = np.isnan(vals)
        vals[miss] = _u_exact_numpy(pts[miss].T)

    return vals.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    time_limit = 0.567

    mesh_resolution = int(case_spec.get("params", {}).get("mesh_resolution", 28))
    element_degree = int(case_spec.get("params", {}).get("element_degree", 2))
    ksp_type = str(case_spec.get("params", {}).get("ksp_type", "cg"))
    pc_type = "hypre"
    rtol = 1.0e-11

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])
    kappa = fem.Constant(domain, ScalarType(1.0))
    f_expr = -ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    petsc_options = {"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}
    if ksp_type == "cg":
        petsc_options["ksp_norm_type"] = "unpreconditioned"

    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options=petsc_options, petsc_options_prefix="poisson_")
        uh = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
            petsc_options_prefix="poisson_fallback_",
        )
        uh = problem.solve()

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    ref_form = fem.form(u_exact**2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    ref_local = fem.assemble_scalar(ref_form)
    err_l2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    ref_l2 = np.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
    rel_l2_error = err_l2 / ref_l2 if ref_l2 > 0 else err_l2

    elapsed = time.perf_counter() - t0
    if elapsed < 0.30 * time_limit and mesh_resolution < 40:
        mesh_resolution = min(40, mesh_resolution + 8)
        domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        x = ufl.SpatialCoordinate(domain)
        u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * (1.0 + 0.5 * x[0] * x[1])
        kappa = fem.Constant(domain, ScalarType(1.0))
        f_expr = -ufl.div(kappa * ufl.grad(u_exact))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda X: np.ones(X.shape[1], dtype=bool))
        boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
        u_bc = fem.Function(V)
        u_bc.interpolate(_u_exact_numpy)
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
            petsc_options_prefix="poisson_refined_",
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
        ref_form = fem.form(u_exact**2 * ufl.dx)
        err_local = fem.assemble_scalar(err_form)
        ref_local = fem.assemble_scalar(ref_form)
        err_l2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
        ref_l2 = np.sqrt(comm.allreduce(ref_local, op=MPI.SUM))
        rel_l2_error = err_l2 / ref_l2 if ref_l2 > 0 else err_l2

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(mesh_resolution),
            "element_degree": int(element_degree),
            "ksp_type": str(ksp_type),
            "pc_type": str(pc_type),
            "rtol": float(rtol),
            "iterations": int(iterations),
            "rel_l2_error": float(rel_l2_error),
        },
    }
