import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse parameters
    pde = case_spec["pde"]
    eps_val = pde["coefficients"]["epsilon"]
    beta_val = pde["coefficients"]["beta"]
    time_params = pde["time"]
    t_end = time_params["t_end"]
    dt_suggested = time_params["dt"]

    # Choose parameters
    N = 80  # mesh resolution
    degree = 1
    dt = dt_suggested  # 0.01
    n_steps = int(round(t_end / dt))

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))

    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])

    # Source term derived from manufactured solution
    # u_t = -exp(-t)*sin(2*pi*x)*sin(pi*y)
    # -eps*laplacian(u) = eps*exp(-t)*(4*pi^2 + pi^2)*sin(2*pi*x)*sin(pi*y)
    # beta . grad(u) = exp(-t)*(beta[0]*2*pi*cos(2*pi*x)*sin(pi*y) + beta[1]*sin(2*pi*x)*pi*cos(pi*y))
    # f = u_t - eps*laplacian(u) + beta.grad(u)

    beta = fem.Constant(domain, np.array(beta_val, dtype=np.float64))
    eps = fem.Constant(domain, ScalarType(eps_val))

    u_t = -ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    lap_u = -ufl.exp(-t_const) * (4 * pi**2 + pi**2) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.as_vector([
        ufl.exp(-t_const) * 2 * pi * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1]),
        ufl.exp(-t_const) * ufl.sin(2 * pi * x[0]) * pi * ufl.cos(pi * x[1])
    ])
    f_expr = u_t - eps * lap_u + ufl.dot(beta, grad_u_exact)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")

    # Initial condition at t=0
    t_const.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * eps)
    # SUPG parameter
    tau = h / (2.0 * beta_norm) * (ufl.cosh(Pe_cell) / ufl.sinh(Pe_cell) - 1.0 / Pe_cell)
    # Simpler: tau = h / (2 * beta_norm) for high Pe
    # Use the xiong formula but clamp:
    # Actually let's use a simpler robust formula
    tau = h / (2.0 * beta_norm + 1e-10)

    # Backward Euler: (u - u_n)/dt - eps*lap(u) + beta.grad(u) = f
    # Weak form (Galerkin part):
    a_gal = (u * v / dt) * ufl.dx + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_gal = (u_n / dt) * v * ufl.dx + f_expr * v * ufl.dx

    # SUPG stabilization terms
    # Residual applied to trial: R(u) = u/dt - eps*lap(u) + beta.grad(u) - f
    # For linear elements, lap(u) = 0 in each cell
    # So residual ≈ u/dt + beta.grad(u) - u_n/dt - f
    R_trial = u / dt + ufl.dot(beta, ufl.grad(u))
    R_rhs = u_n / dt + f_expr

    # SUPG test function modification: v_supg = tau * beta . grad(v)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    a_supg = a_gal + R_trial * v_supg * ufl.dx
    L_supg = L_gal + R_rhs * v_supg * ufl.dx

    # Boundary conditions (all boundary)
    def boundary(x_arr):
        return (np.isclose(x_arr[0], 0.0) | np.isclose(x_arr[0], 1.0) |
                np.isclose(x_arr[1], 0.0) | np.isclose(x_arr[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bc_dofs)

    # Compile forms
    a_form = fem.form(a_supg)
    L_form = fem.form(L_supg)

    # Assemble matrix (constant in time since coefficients don't change except t in f)
    # Actually, the bilinear form doesn't depend on t, so A is constant
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = fem.petsc.create_vector(L_form)

    # Solution function
    u_sol = fem.Function(V)

    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setUp()

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t_val = (step + 1) * dt
        t_const.value = t_val

        # Update BC
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }