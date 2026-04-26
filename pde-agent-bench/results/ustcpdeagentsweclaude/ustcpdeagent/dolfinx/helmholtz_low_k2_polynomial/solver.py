import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


def solve(case_spec: dict) -> dict:
    # ```DIAGNOSIS
    # equation_type:        helmholtz
    # spatial_dim:          2
    # domain_geometry:      rectangle
    # unknowns:             scalar
    # coupling:             none
    # linearity:            linear
    # time_dependence:      steady
    # stiffness:            N/A
    # dominant_physics:     wave
    # peclet_or_reynolds:   N/A
    # solution_regularity:  smooth
    # bc_type:              all_dirichlet
    # special_notes:        manufactured_solution
    # ```
    # ```METHOD
    # spatial_method:       fem
    # element_or_basis:     Lagrange_P2
    # stabilization:        none
    # time_method:          none
    # nonlinear_solver:     none
    # linear_solver:        gmres
    # preconditioner:       ilu
    # special_treatment:    none
    # pde_skill:            helmholtz
    # ```

    comm = MPI.COMM_WORLD
    k = float(case_spec.get("pde", {}).get("k", case_spec.get("pde", {}).get("wavenumber", 2.0)))
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    # Choose P2 on a moderate mesh for high accuracy within tight time budget.
    mesh_resolution = 24
    element_degree = 2
    rtol = 1e-10
    ksp_type = "gmres"
    pc_type = "ilu"

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    lap_u_exact = -2.0 * x[1] * (1.0 - x[1]) - 2.0 * x[0] * (1.0 - x[0])
    f_ufl = -lap_u_exact - k**2 * u_exact_ufl

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    uh = fem.Function(V)
    iterations = 0

    try:
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix="helmholtz_",
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": rtol,
                "ksp_atol": 1e-14,
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())
        ksp_type = problem.solver.getType()
        pc_type = problem.solver.getPC().getType()
    except Exception:
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix="helmholtz_fallback_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        iterations = int(problem.solver.getIterationNumber())
        ksp_type = problem.solver.getType()
        pc_type = problem.solver.getPC().getType()

    # Accuracy verification against manufactured solution
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_sq_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_l2 = np.sqrt(comm.allreduce(err_sq_local, op=MPI.SUM))

    # Sample onto required uniform grid
    xs = np.linspace(xmin, xmax, nx_out, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny_out, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
    eval_pts = []
    eval_cells = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_pts:
        vals = uh.eval(np.array(eval_pts, dtype=np.float64), np.array(eval_cells, dtype=np.int32)).reshape(-1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.real_if_close(vals)

    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        vals = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(vals) & ~np.isnan(arr)
            vals[mask] = arr[mask]
        u_grid = vals.reshape((ny_out, nx_out))
        result = {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": mesh_resolution,
                "element_degree": element_degree,
                "ksp_type": str(ksp_type),
                "pc_type": str(pc_type),
                "rtol": float(rtol),
                "iterations": int(iterations),
                "l2_error": float(err_l2),
            },
        }
    else:
        result = {"u": None, "solver_info": None}

    return result
