from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def _sample_on_grid(domain, u_fun, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
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
        vals = u_fun.eval(np.asarray(points_on_proc, dtype=np.float64),
                          np.asarray(cells_on_proc, dtype=np.int32))
        local_vals[np.asarray(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        return out.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    """
    Solve Δ²u = f on [0,1]x[0,1] using a mixed formulation:
        w = -Δu
        -Δw = f
    with manufactured exact solution
        u = tanh(6*(y-0.5))*sin(pi*x)

    Returns:
      {"u": u_grid, "solver_info": {...}}
    """
    comm = MPI.COMM_WORLD

    # ```DIAGNOSIS
    # equation_type: biharmonic
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar+scalar
    # coupling: sequential
    # linearity: linear
    # time_dependence: steady
    # stiffness: stiff
    # dominant_physics: diffusion
    # peclet_or_reynolds: N/A
    # solution_regularity: boundary_layer
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```

    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: cg
    # preconditioner: hypre
    # special_treatment: problem_splitting
    # pde_skill: none
    # ```

    # Accuracy/time tuned for the given benchmark budget
    mesh_resolution = 96
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1.0e-10

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    s = 6.0 * (x[1] - 0.5)
    t = ufl.tanh(s)
    sech2 = 1.0 / ufl.cosh(s) ** 2

    u_exact = t * ufl.sin(pi * x[0])

    # Derivatives for tanh(6(y-0.5))
    T2 = -72.0 * sech2 * t
    T4 = 5184.0 * sech2 * t * (2.0 - 3.0 * sech2)

    lap_u = (T2 - pi**2 * t) * ufl.sin(pi * x[0])
    f_expr = (T4 - 2.0 * pi**2 * T2 + pi**4 * t) * ufl.sin(pi * x[0])
    w_bc_expr = -lap_u

    # Boundary conditions
    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    w_bc_fun = fem.Function(V)
    w_bc_fun.interpolate(fem.Expression(w_bc_expr, V.element.interpolation_points))

    f_fun = fem.Function(V)
    f_fun.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, bfacets)

    bc_u = fem.dirichletbc(u_bc_fun, bdofs)
    bc_w = fem.dirichletbc(w_bc_fun, bdofs)

    # First Poisson solve: -Δw = f
    w_tr = ufl.TrialFunction(V)
    z = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(w_tr), ufl.grad(z)) * ufl.dx
    L1 = ufl.inner(f_fun, z) * ufl.dx

    prob1 = petsc.LinearProblem(
        a1,
        L1,
        bcs=[bc_w],
        petsc_options_prefix="biharmonic_w_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )
    w_h = prob1.solve()
    w_h.x.scatter_forward()

    # Second Poisson solve: -Δu = w
    u_tr = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a2 = ufl.inner(ufl.grad(u_tr), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_h, v) * ufl.dx

    prob2 = petsc.LinearProblem(
        a2,
        L2,
        bcs=[bc_u],
        petsc_options_prefix="biharmonic_u_",
        petsc_options={
            "ksp_type": ksp_type,
            "ksp_rtol": rtol,
            "pc_type": pc_type,
        },
    )
    u_h = prob2.solve()
    u_h.x.scatter_forward()

    # Accuracy verification against exact manufactured solution
    u_ex_fun = fem.Function(V)
    u_ex_fun.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    e2 = fem.assemble_scalar(fem.form((u_h - u_ex_fun) ** 2 * ufl.dx))
    n2 = fem.assemble_scalar(fem.form((u_ex_fun) ** 2 * ufl.dx))
    e2 = comm.allreduce(e2, op=MPI.SUM)
    n2 = comm.allreduce(n2, op=MPI.SUM)
    rel_l2_error = float(np.sqrt(e2 / max(n2, 1.0e-30)))

    # Iteration counts (best-effort)
    iterations = 0
    try:
        iterations += int(prob1.solver.getIterationNumber())
    except Exception:
        pass
    try:
        iterations += int(prob2.solver.getIterationNumber())
    except Exception:
        pass

    u_grid = _sample_on_grid(domain, u_h, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "verification_rel_l2_error": rel_l2_error,
    }

    return {"u": u_grid, "solver_info": solver_info}
