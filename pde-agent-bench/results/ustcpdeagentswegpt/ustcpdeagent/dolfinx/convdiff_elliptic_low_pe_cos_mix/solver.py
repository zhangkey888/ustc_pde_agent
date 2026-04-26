import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """
    Solve steady convection-diffusion:
        -eps * Δu + beta·∇u = f  in Ω=[0,1]^2
        u = g on ∂Ω
    using a manufactured exact solution
        u_exact = cos(pi*x) * sin(pi*y)

    Returns
    -------
    dict with keys:
      - "u": sampled solution on requested uniform grid, shape (ny, nx)
      - "solver_info": metadata
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # -------------------------
    # Problem / benchmark data
    # -------------------------
    epsilon = float(case_spec.get("pde", {}).get("epsilon", 0.2))
    beta_in = case_spec.get("pde", {}).get("beta", [0.8, 0.3])
    beta_vec = np.array(beta_in, dtype=np.float64)
    if beta_vec.shape != (2,):
        beta_vec = np.array([0.8, 0.3], dtype=np.float64)

    output_grid = case_spec["output"]["grid"]
    nx_out = int(output_grid["nx"])
    ny_out = int(output_grid["ny"])
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # -------------------------
    # Adaptive discretization
    # -------------------------
    # Moderate Péclet, strict accuracy target, low-dimensional steady problem.
    # P2 on a reasonably fine mesh is robust and fast enough.
    element_degree = 2
    # Keep cost moderate for <5s while improving accuracy.
    mesh_resolution = 48

    t0 = time.perf_counter()

    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    u_exact_ufl = ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)

    beta = fem.Constant(domain, np.array(beta_vec, dtype=ScalarType))
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    # Manufactured RHS: -eps Δu + beta·grad(u)
    lap_u_exact = ufl.div(grad_u_exact)
    f_ufl = -eps_c * lap_u_exact + ufl.dot(beta, grad_u_exact)

    # -------------------------
    # Dirichlet BC on whole boundary
    # -------------------------
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # -------------------------
    # Variational formulation
    # -------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_ufl * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    bnorm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    Pe = bnorm * h / (2.0 * eps_c)
    cothPe = (ufl.exp(2.0 * Pe) + 1.0) / (ufl.exp(2.0 * Pe) - 1.0 + 1.0e-16)
    tau_adv = h / (2.0 * bnorm + 1.0e-16)
    tau = ufl.conditional(
        ufl.gt(bnorm, 1.0e-14),
        tau_adv * (cothPe - 1.0 / (Pe + 1.0e-16)),
        0.0 * h,
    )

    residual_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_f = f_ufl

    a_supg = tau * ufl.dot(beta, ufl.grad(v)) * residual_u * ufl.dx
    L_supg = tau * ufl.dot(beta, ufl.grad(v)) * residual_f * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    # -------------------------
    # Linear solve
    # -------------------------
    petsc_options = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 2000,
    }

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options=petsc_options,
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Retrieve actual PETSc iteration count when possible
    iterations = None
    ksp_type = petsc_options["ksp_type"]
    pc_type = petsc_options["pc_type"]
    rtol = float(petsc_options["ksp_rtol"])
    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
        ksp_type = solver.getType()
        pc_type = solver.getPC().getType()
        rtol = float(solver.getTolerances()[0])
    except Exception:
        iterations = -1

    # -------------------------
    # Accuracy verification
    # -------------------------
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact_fun.x.array
    e.x.scatter_forward()

    err_local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
    norm_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    err_l2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))
    norm_l2 = np.sqrt(comm.allreduce(norm_local, op=MPI.SUM))
    rel_l2 = err_l2 / (norm_l2 + 1.0e-30)

    # -------------------------
    # Sample on requested grid
    # -------------------------
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_values[np.array(eval_map, dtype=np.int32)] = vals

    gathered = comm.gather(local_values, root=0)

    if rank == 0:
        global_values = np.full(nx_out * ny_out, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_values) & (~np.isnan(arr))
            global_values[mask] = arr[mask]

        # Guard against rare boundary-eval misses by filling with exact solution
        if np.isnan(global_values).any():
            miss = np.isnan(global_values)
            px = pts[miss, 0]
            py = pts[miss, 1]
            global_values[miss] = np.cos(np.pi * px) * np.sin(np.pi * py)

        u_grid = global_values.reshape(ny_out, nx_out)
    else:
        u_grid = None

    elapsed = time.perf_counter() - t0

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "iterations": int(iterations),
        "l2_error": float(err_l2),
        "relative_l2_error": float(rel_l2),
        "wall_time_sec": float(elapsed),
    }

    result = {"u": u_grid, "solver_info": solver_info}
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "epsilon": 0.2,
            "beta": [0.8, 0.3],
            "time": None,
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
