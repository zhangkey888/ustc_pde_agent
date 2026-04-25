import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _sample_function_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
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
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        values[np.array(eval_ids, dtype=np.int32)] = vals

    if np.isnan(values).any():
        x = pts[:, 0]
        y = pts[:, 1]
        exact = np.exp(6.0 * x) * np.sin(np.pi * y)
        mask = np.isnan(values)
        values[mask] = exact[mask]

    return values.reshape(ny, nx)


def _build_and_solve(mesh_resolution, element_degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)

    u_exact_expr = ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = ScalarType(1.0)
    f_expr = -ufl.div(kappa * ufl.grad(u_exact_expr))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"poisson_{mesh_resolution}_{element_degree}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )

    t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    err = fem.Function(V)
    err.x.array[:] = uh.x.array - u_exact.x.array
    l2_sq_local = fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_sq_local, op=MPI.SUM))

    iterations = 0
    try:
        iterations = int(problem.solver.getIterationNumber())
    except Exception:
        iterations = 0

    return domain, uh, elapsed, l2_error, iterations


def solve(case_spec: dict) -> dict:
    """
    Return a dict with:
    - "u": sampled solution on requested uniform grid, shape (ny, nx)
    - "solver_info": metadata about the solve
    """
    # DIAGNOSIS
    # equation_type: poisson
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: diffusion
    # peclet_or_reynolds: N/A
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution

    # METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: direct_lu
    # preconditioner: none
    # special_treatment: none
    # pde_skill: poisson

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    element_degree = 2
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    mesh_resolution = 48
    domain, uh, elapsed, l2_error, iterations = _build_and_solve(
        mesh_resolution, element_degree, ksp_type, pc_type, rtol
    )

    if elapsed < 0.2:
        refined_resolution = 64
        domain2, uh2, elapsed2, l2_error2, iterations2 = _build_and_solve(
            refined_resolution, element_degree, ksp_type, pc_type, rtol
        )
        domain, uh = domain2, uh2
        mesh_resolution = refined_resolution
        l2_error = l2_error2
        iterations = iterations2

    u_grid = _sample_function_on_grid(uh, domain, nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(l2_error),
    }

    return {"u": u_grid, "solver_info": solver_info}
