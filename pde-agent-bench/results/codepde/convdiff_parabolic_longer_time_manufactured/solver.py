import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.05)
    beta = params.get("beta", [2.0, 1.0])
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.2)
    dt_suggested = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")

    # Use a finer mesh and smaller dt for accuracy
    N = 80
    degree = 1
    dt = 0.005  # smaller than suggested for accuracy
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # exact division

    comm = MPI.COMM_WORLD

    # 2. Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # 4. Manufactured solution: u_exact = exp(-2*t)*sin(pi*x)*sin(pi*y)
    # Source term: f = du/dt - eps*laplacian(u) + beta . grad(u)
    # du/dt = -2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # laplacian(u) = -2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    # grad(u) = exp(-2*t)*[pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    # f = -2*exp(-2*t)*sin(pi*x)*sin(pi*y) + eps*2*pi^2*exp(-2*t)*sin(pi*x)*sin(pi*y)
    #     + beta[0]*exp(-2*t)*pi*cos(pi*x)*sin(pi*y) + beta[1]*exp(-2*t)*pi*sin(pi*x)*cos(pi*y)

    t_const = fem.Constant(domain, default_scalar_type(0.0))
    beta_vec = fem.Constant(domain, np.array(beta, dtype=default_scalar_type))
    eps_const = fem.Constant(domain, default_scalar_type(epsilon))
    dt_const = fem.Constant(domain, default_scalar_type(dt))

    # Exact solution as UFL expression (for source term and BCs)
    def u_exact_ufl(t_val):
        return ufl.exp(-2 * t_val) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term at time t
    def f_ufl(t_val):
        u_ex = ufl.exp(-2 * t_val) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        dudt = -2.0 * u_ex
        lap_u = -2.0 * pi**2 * u_ex
        grad_u = ufl.grad(u_ex)
        return dudt - epsilon * lap_u + ufl.dot(ufl.as_vector([beta[0], beta[1]]), grad_u)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)

    # Initial condition
    u_n.interpolate(lambda x_arr: np.exp(-2.0 * 0.0) * np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))

    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    # Point evaluation setup
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.zeros((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.zeros(0, dtype=np.int32)

    def evaluate_on_grid(u_func):
        u_values = np.full(points_2d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(points_on_proc_arr, cells_on_proc_arr)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)

    u_initial = evaluate_on_grid(u_n)

    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_ufl = ufl.as_vector([beta[0], beta[1]])
    beta_norm = ufl.sqrt(ufl.dot(beta_ufl, beta_ufl))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    # tau_supg = h / (2 * |beta|) * (coth(Pe) - 1/Pe) ~ for high Pe ~ h/(2|beta|)
    tau_supg = h / (2.0 * beta_norm) * (1.0 - 1.0 / Pe_cell)
    # Clamp to non-negative (approximate)
    # For high Pe this is fine

    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f(t_{n+1})
    # Weak form: (u - u_n)/dt * v + eps*grad(u).grad(v) + (beta.grad(u))*v = f*v
    # With SUPG: add tau_supg * (residual) * (beta.grad(v)) over each cell

    # f at current time (will be updated via t_const)
    f_expr = f_ufl(t_const)

    # Standard Galerkin bilinear form
    a_std = (u * v / dt_const) * ufl.dx \
            + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.dot(beta_ufl, ufl.grad(u)) * v * ufl.dx

    L_std = (u_n * v / dt_const) * ufl.dx + f_expr * v * ufl.dx

    # SUPG terms
    # Residual of the strong form applied to trial function:
    # R(u) = u/dt - eps*lap(u) + beta.grad(u) - (u_n/dt + f)
    # For the bilinear part: u/dt + beta.grad(u)  (we skip -eps*lap(u) for linear elements since lap=0)
    # For linear elements on triangles, laplacian of u_h = 0 within each cell
    # So strong residual bilinear: u/dt + beta.grad(u)
    # Strong residual rhs: u_n/dt + f

    supg_test = tau_supg * ufl.dot(beta_ufl, ufl.grad(v))

    a_supg = (u / dt_const + ufl.dot(beta_ufl, ufl.grad(u))) * supg_test * ufl.dx
    L_supg = (u_n / dt_const + f_expr) * supg_test * ufl.dx

    a_form = a_std + a_supg
    L_form = L_std + L_supg

    # 5. Boundary conditions - Dirichlet on all boundaries
    # g = u_exact at current time, interpolated each step
    u_bc_func = fem.Function(V)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc = fem.dirichletbc(u_bc_func, dofs)

    # 6. Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble and solve with manual assembly for efficiency in time loop
    u_sol = fem.Function(V)

    # Setup KSP solver
    solver = PETSc.KSP().create(comm)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iterations = 0

    current_t = 0.0
    for step in range(n_steps):
        current_t += dt
        t_const.value = current_t

        # Update boundary condition
        t_val = current_t
        u_bc_func.interpolate(
            lambda x_arr, t_v=t_val: np.exp(-2.0 * t_v) * np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1])
        )

        # Assemble matrix
        A = petsc.assemble_matrix(a_compiled, bcs=[bc])
        A.assemble()

        # Assemble RHS
        b = petsc.create_vector(L_compiled)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.setOperators(A)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

        # Clean up PETSc objects
        A.destroy()
        b.destroy()

    # 7. Extract on grid
    u_grid = evaluate_on_grid(u_sol)

    solver.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }