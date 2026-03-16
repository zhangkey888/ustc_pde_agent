import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nx = ny = 64
    degree = 2
    t_end = 0.4
    dt_val = 0.01
    n_steps = int(round(t_end / dt_val))

    # Create quadrilateral mesh on [0,1]x[0,1]
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    eps_val = 1.0  # diffusion coefficient (default for reaction-diffusion)

    # Manufactured solution: u = exp(-t)*(exp(x)*sin(pi*y))
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * (ufl.exp(x[0]) * ufl.sin(pi * x[1]))

    # Reaction term R(u) = u (linear reaction for simplicity, adjust source accordingly)
    # We need to derive f from: du/dt - eps*laplacian(u) + R(u) = f
    # with R(u) = u (linear reaction)
    # u = exp(-t)*exp(x)*sin(pi*y)
    # du/dt = -exp(-t)*exp(x)*sin(pi*y) = -u
    # laplacian(u) = exp(-t)*(exp(x)*sin(pi*y) - pi^2*exp(x)*sin(pi*y)) = u*(1 - pi^2)
    # -eps*laplacian(u) = -eps*u*(1 - pi^2) = eps*u*(pi^2 - 1)
    # R(u) = u
    # f = du/dt - eps*laplacian(u) + R(u) = -u + eps*u*(pi^2-1) + u = eps*u*(pi^2-1)
    # f = eps*(pi^2 - 1)*u_exact
    # But let's compute it symbolically to be safe

    # Actually, let me compute the source term properly using UFL differentiation
    # du/dt = -u_exact (since u = exp(-t)*g(x,y))
    # grad(u) = exp(-t)*(exp(x)*sin(pi*y), exp(x)*pi*cos(pi*y))
    # laplacian(u) = exp(-t)*(exp(x)*sin(pi*y) + exp(x)*(-pi^2)*sin(pi*y))
    #              = exp(-t)*exp(x)*sin(pi*y)*(1 - pi^2)
    #              = u_exact*(1 - pi^2)

    # For R(u) = u:
    # f = du/dt - eps*laplacian(u) + R(u)
    # f = -u_exact - eps*u_exact*(1-pi^2) + u_exact
    # f = -u_exact - eps*u_exact + eps*pi^2*u_exact + u_exact
    # f = eps*(pi^2 - 1)*u_exact

    f_expr = ufl.exp(-t) * ufl.exp(x[0]) * ufl.sin(pi * x[1]) * (
        -1.0 - eps_val * (1.0 - pi**2) + 1.0
    )
    # Simplify: = u_exact * (-1 - eps*(1-pi^2) + 1) = u_exact * (eps*(pi^2-1))
    # Let me just write it directly:
    f_expr = eps_val * (pi**2 - 1.0) * ufl.exp(-t) * ufl.exp(x[0]) * ufl.sin(pi * x[1])

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # current solution (for Newton)
    v = ufl.TestFunction(V)

    # Reaction term R(u) = u (linear)
    R_u = u_h

    # Backward Euler: (u_h - u_n)/dt - eps*laplacian(u_h) + R(u_h) = f
    # Weak form (residual):
    F = (
        ufl.inner((u_h - u_n) / dt, v) * ufl.dx
        + eps_val * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
        + ufl.inner(R_u, v) * ufl.dx
        - ufl.inner(f_expr, v) * ufl.dx
    )

    # Boundary conditions: u = u_exact on all boundaries
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc_func = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Expression for exact solution
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # Set initial condition at t=0
    t.value = 0.0
    u_bc_func.interpolate(u_exact_expr)
    u_n.interpolate(u_exact_expr)
    u_h.x.array[:] = u_n.x.array[:]

    bc = fem.dirichletbc(u_bc_func, bc_dofs)
    bcs = [bc]

    # Since R(u) = u is linear, the problem is actually linear
    # But we'll use Newton solver for generality (it converges in 1 iteration for linear problems)
    # Actually, let's use a linear solve approach for efficiency

    # Rewrite as linear problem:
    # (u - u_n)/dt - eps*laplacian(u) + u = f
    # => u/dt + eps*(-laplacian(u)) + u = f + u_n/dt
    # Bilinear form:
    u_trial = ufl.TrialFunction(V)
    a = (
        ufl.inner(u_trial / dt, v) * ufl.dx
        + eps_val * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        + ufl.inner(u_trial, v) * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx + ufl.inner(u_n / dt, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time since coefficients don't change)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(L_form)

    # Set up KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()

    total_iterations = 0

    # Store initial condition for output
    u_initial_grid = None

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val

        # Update BC
        u_bc_func.interpolate(u_exact_expr)

        # Reassemble matrix (BCs might change lifting)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate on 60x60 grid
    nx_eval, ny_eval = 60, 60
    xs = np.linspace(0.0, 1.0, nx_eval)
    ys = np.linspace(0.0, 1.0, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_eval * ny_eval))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()

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

    u_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_eval, ny_eval))

    # Also compute initial condition on same grid
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)

    u_init_values = np.full(nx_eval * ny_eval, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((nx_eval, ny_eval))

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }