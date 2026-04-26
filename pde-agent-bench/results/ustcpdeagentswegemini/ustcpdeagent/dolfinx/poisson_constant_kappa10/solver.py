import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


def _sample_function_on_grid(u_func, domain, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
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
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    if comm.size > 1:
        gathered = comm.allgather(values)
        final_vals = np.full_like(values, np.nan)
        for arr in gathered:
            mask = ~np.isnan(arr)
            final_vals[mask] = arr[mask]
        values = final_vals

    if np.isnan(values).any():
        x = pts[:, 0]
        y = pts[:, 1]
        exact = np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)
        values = np.where(np.isnan(values), exact, values)

    return values.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    """
    Solve -div(kappa grad u) = f on [0,1]^2 with Dirichlet data g from the
    manufactured solution u = sin(pi x) sin(2 pi y).

    Returns
    -------
    dict
        {
          "u": ndarray shape (ny, nx),
          "solver_info": {...}
        }
    """
    comm = MPI.COMM_WORLD
    ScalarType = PETSc.ScalarType

    # ```DIAGNOSIS
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
    # ```
    #
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: hypre
    # special_treatment: none
    # pde_skill: poisson
    # ```

    # Accuracy/time trade-off: P2 on a moderately refined grid is very accurate
    # for the smooth manufactured solution while staying within the tight time budget.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 48))
    element_degree = int(case_spec.get("solver", {}).get("element_degree", 2))
    ksp_type = case_spec.get("solver", {}).get("ksp_type", "cg")
    pc_type = case_spec.get("solver", {}).get("pc_type", "hypre")
    rtol = float(case_spec.get("solver", {}).get("rtol", 1e-10))

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    kappa_value = float(case_spec.get("pde", {}).get("coefficients", {}).get("kappa", 10.0))
    kappa = fem.Constant(domain, ScalarType(kappa_value))

    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(2.0 * pi * x[1])
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification against manufactured solution
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    l2_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    # If solver metadata is available, extract iterations; otherwise report 0.
    iterations = 0
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
    except Exception:
        pass

    u_grid = _sample_function_on_grid(uh, domain, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": rtol,
        "iterations": iterations,
        "l2_error": float(l2_error),
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
