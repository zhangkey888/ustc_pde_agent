import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _g_callable(x):
    return np.sin(np.pi * x[0]) + np.cos(np.pi * x[1])


def _all_boundary(x):
    return np.ones(x.shape[1], dtype=bool)


def _sample_function(u_func, pts):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.asarray(points_on_proc, dtype=np.float64),
            np.asarray(cells_on_proc, dtype=np.int32),
        )
        values[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    return values


def _solve_poisson(comm, mesh_resolution, element_degree, ksp_type, pc_type, rtol):
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_bc = fem.Function(V)
    u_bc.interpolate(_g_callable)

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, _all_boundary)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    kappa = fem.Constant(domain, ScalarType(1.0))
    f = fem.Constant(domain, ScalarType(1.0))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    opts = {"ksp_type": ksp_type, "pc_type": pc_type}

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{mesh_resolution}_{element_degree}_",
        petsc_options=opts,
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    z = ufl.TestFunction(V)
    residual_form = ufl.inner(ufl.grad(uh), ufl.grad(z)) * ufl.dx - ufl.inner(f, z) * ufl.dx
    res_vec = petsc.create_vector(fem.form(residual_form).function_spaces)
    with res_vec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(res_vec, fem.form(residual_form))
    petsc.set_bc(res_vec, [fem.dirichletbc(ScalarType(0.0), dofs, V)])
    residual_indicator = float(res_vec.norm())

    return domain, uh, iterations, residual_indicator


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    mesh_resolution = 96
    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    try:
        domain, uh, iterations, residual_indicator = _solve_poisson(
            comm, mesh_resolution, element_degree, ksp_type, pc_type, rtol
        )
    except Exception:
        mesh_resolution = 72
        element_degree = 2
        ksp_type = "cg"
        pc_type = "hypre"
        rtol = 1.0e-10
        domain, uh, iterations, residual_indicator = _solve_poisson(
            comm, mesh_resolution, element_degree, ksp_type, pc_type, rtol
        )

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    sampled = _sample_function(uh, pts)
    nan_mask = np.isnan(sampled)
    if np.any(nan_mask):
        sampled[nan_mask] = _g_callable(
            np.vstack([pts[nan_mask, 0], pts[nan_mask, 1], pts[nan_mask, 2]])
        )
    u_grid = sampled.reshape(ny, nx)

    verification = {}
    elapsed = time.perf_counter() - t0
    if elapsed < 6.5:
        try:
            ref_resolution = min(2 * mesh_resolution, 160)
            _, uref, ref_iterations, _ = _solve_poisson(
                comm, ref_resolution, element_degree, "preonly", "lu", 1.0e-12
            )
            u_vals = sampled.copy()
            uref_vals = _sample_function(uref, pts)
            ref_nan = np.isnan(uref_vals)
            if np.any(ref_nan):
                uref_vals[ref_nan] = _g_callable(
                    np.vstack([pts[ref_nan, 0], pts[ref_nan, 1], pts[ref_nan, 2]])
                )
            verification = {
                "reference_mesh_resolution": int(ref_resolution),
                "reference_iterations": int(ref_iterations),
                "grid_max_abs_error_vs_ref": float(np.max(np.abs(u_vals - uref_vals))),
                "grid_rms_error_vs_ref": float(np.sqrt(np.mean((u_vals - uref_vals) ** 2))),
            }
        except Exception:
            verification = {"reference_mesh_resolution": None}

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification_cell_residual_l2_sq": float(residual_indicator),
    }
    solver_info.update(verification)

    return {"u": u_grid, "solver_info": solver_info}
