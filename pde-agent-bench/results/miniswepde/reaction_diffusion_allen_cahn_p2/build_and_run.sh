cat <<'EOF' > solver.py
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl


def _manufactured_solution_expr(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * 0.25 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def _allen_cahn_reaction(u):
    # Standard Allen–Cahn: R(u) = u^3 - u
    return u**3 - u


def _sample_on_grid(domain, u_func, nx, ny):
    """
    Sample scalar Function u_func on a uniform [0,1]x[0,1] grid of size (nx, ny).
    Returns numpy array of shape (nx, ny).
    """
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.zeros((3, nx * ny), dtype=np.float64)
    pts[0, :] = X.ravel()
    pts[1, :] = Y.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((pts.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        points_arr = np.array(points_on_proc, dtype=np.float64)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(points_arr, cells_arr)
        values[np.array(eval_map, dtype=int)] = vals.flatten()

    return values.reshape((nx, ny))


def _gather_grid(comm, local_grid):
    """
    Gather grid data from all ranks to rank 0.
    Assumes all ranks have same shape.
    """
    rank = comm.rank
    if local_grid is None:
        local_flat = np.array([], dtype=np.float64)
    else:
        local_flat = np.asarray(local_grid, dtype=np.float64).ravel()

    counts = comm.gather(len(local_flat), root=0)
    if rank == 0:
        if counts is None:
            return local_grid
        recvbuf = np.empty(sum(counts), dtype=np.float64)
        displs = np.insert(np.cumsum(counts), 0, 0)[0:-1]
        comm.Gatherv(local_flat, [recvbuf, counts, displs, MPI.DOUBLE], root=0)
        if local_grid is None:
            return None
        global_flat = recvbuf[: len(local_flat)]
        return global_flat.reshape(local_grid.shape)
    else:
        comm.Gatherv(local_flat, None, root=0)
        return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Parameters (with sensible defaults)
    pde_spec = case_spec.get("pde", {})
    time_spec = pde_spec.get("time", {})
    t_end = float(time_spec.get("t_end", 0.2))
    dt = float(time_spec.get("dt", 0.005))
    time_scheme = time_spec.get("scheme", "backward_euler")

    mesh_resolution = int(case_spec.get("mesh_resolution", 64))
    element_degree = int(case_spec.get("element_degree", 2))
    eps = float(case_spec.get("epsilon", 0.01))  # diffusion coefficient

    # Create mesh
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Time stepping info
    t = 0.0
    n_steps = int(round(t_end / dt))

    # Manufactured solution and initial condition
    u_exact_expr_t0 = _manufactured_solution_expr(domain, t)
    u0_expr = fem.Expression(u_exact_expr_t0, V.element.interpolation_points)
    u_n = fem.Function(V)
    u_n.interpolate(u0_expr)

    # Store initial condition on grid later
    u_initial = None

    # Boundary condition: Dirichlet with exact solution at current time
    fdim = domain.topology.dim - 1

    def boundary_all(x):
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], 1.0),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], 1.0),
            )
        )

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    bdry_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc_fun = fem.Function(V)

    def update_bc(t_local):
        u_exact_expr = _manufactured_solution_expr(domain, t_local)
        expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
        u_bc_fun.interpolate(expr)

    update_bc(t)
    bc = fem.dirichletbc(u_bc_fun, bdry_dofs)

    # Unknown and test function
    u = fem.Function(V)  # current iterate / solution
    v = ufl.TestFunction(V)

    # Time-dependent constant for manufactured solution
    t_sym = fem.Constant(domain, PETSc.ScalarType(t))

    # Manufactured source term f from exact solution:
    # u_t - eps Δu + R(u) = f
    x = ufl.SpatialCoordinate(domain)
    u_exact = _manufactured_solution_expr(domain, t_sym)
    u_t = -u_exact  # since u = exp(-t)*phi(x,y)
    laplace_u = ufl.div(ufl.grad(u_exact))
    f_expr = u_t - eps * laplace_u + _allen_cahn_reaction(u_exact)

    F = (
        (u - u_n) / dt * v * ufl.dx
        + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + _allen_cahn_reaction(u) * v * ufl.dx
        - f_expr * v * ufl.dx
    )

    problem = petsc.NonlinearProblem(F, u, bcs=[bc])
    newton_solver = nls.petsc.NewtonSolver(domain.comm, problem)
    newton_solver.convergence_criterion = "incremental"
    newton_solver.rtol = 1e-8
    newton_solver.atol = 1e-10
    newton_solver.max_it = 25

    # Configure linear solver inside Newton
    ksp = newton_solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-8)

    nonlinear_iterations = []
    total_linear_iterations = 0

    def get_ksp_its():
        its = ksp.getIterationNumber()
        return int(its) if its is not None else 0

    # Initialize u with initial condition
    u.x.array[:] = u_n.x.array

    # Time loop
    for step in range(1, n_steps + 1):
        t = step * dt
        t_sym.value = PETSc.ScalarType(t)
        update_bc(t)

        # Initial guess: previous solution
        u.x.array[:] = u_n.x.array

        n_it, converged = newton_solver.solve(u)
        u.x.scatter_forward()
        nonlinear_iterations.append(int(n_it))
        total_linear_iterations += get_ksp_its()

        # Update previous solution
        u_n.x.array[:] = u.x.array

        if step == 1:
            u_initial = _sample_on_grid(domain, u_n, 60, 60)

    # Final solution sampling
    u_grid = _sample_on_grid(domain, u, 60, 60)

    if u_initial is None:
        u_initial = _sample_on_grid(domain, u_n, 60, 60)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": int(total_linear_iterations),
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
        "nonlinear_iterations": nonlinear_iterations,
    }

    u_grid_global = _gather_grid(comm, u_grid)
    u_initial_global = _gather_grid(comm, u_initial)

    if rank == 0:
        return {"u": u_grid_global, "u_initial": u_initial_global, "solver_info": solver_info}
    else:
        return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {
                "t_end": 0.2,
                "dt": 0.005,
                "scheme": "backward_euler",
            }
        },
        "mesh_resolution": 40,
        "element_degree": 2,
        "epsilon": 0.01,
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        u = result["u"]
        print("u shape:", None if u is None else u.shape)
        print("solver_info:", result["solver_info"])
EOF

python solver.py
