import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.6)
    dt_suggested = time_params.get("dt", 0.01)
    scheme = time_params.get("scheme", "backward_euler")

    # Parameters
    mesh_resolution = 80
    element_degree = 2
    dt = dt_suggested
    epsilon = 1.0  # default diffusion coefficient

    # Check if epsilon is specified
    if "coefficients" in pde:
        epsilon = pde["coefficients"].get("epsilon", 1.0)

    # Check for reaction coefficient
    reaction_alpha = 1.0
    if "coefficients" in pde:
        reaction_alpha = pde["coefficients"].get("reaction_alpha", 1.0)

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Spatial coordinates and time
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Manufactured solution: u = exp(-t)*(cos(2*pi*x)*sin(pi*y))
    # We need to derive the source term f
    # u_exact = exp(-t) * cos(2*pi*x) * sin(pi*y)
    # du/dt = -exp(-t) * cos(2*pi*x) * sin(pi*y) = -u
    # nabla^2 u = exp(-t) * [-(2*pi)^2 * cos(2*pi*x)*sin(pi*y) - pi^2 * cos(2*pi*x)*sin(pi*y)]
    #           = exp(-t) * cos(2*pi*x)*sin(pi*y) * [-(4*pi^2 + pi^2)]
    #           = -5*pi^2 * u
    # So: du/dt - epsilon * nabla^2 u + alpha * u = f
    #     -u - epsilon * (-5*pi^2 * u) + alpha * u = f
    #     -u + 5*epsilon*pi^2*u + alpha*u = f
    #     u * (-1 + 5*epsilon*pi^2 + alpha) = f

    # Time parameter as a Constant
    t_const = fem.Constant(domain, ScalarType(0.0))

    # Build UFL expression for exact solution at current time
    u_exact_ufl = ufl.exp(-t_const) * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1])

    # Source term derived from manufactured solution
    # f = du/dt - eps * laplacian(u) + alpha * u
    # du/dt = -exp(-t)*cos(2*pi*x)*sin(pi*y)
    # -eps * laplacian(u) = eps * 5*pi^2 * exp(-t)*cos(2*pi*x)*sin(pi*y)
    # alpha * u = alpha * exp(-t)*cos(2*pi*x)*sin(pi*y)
    # f = exp(-t)*cos(2*pi*x)*sin(pi*y) * (-1 + 5*eps*pi^2 + alpha)
    f_ufl = ufl.exp(-t_const) * ufl.cos(2 * pi * x[0]) * ufl.sin(pi * x[1]) * (
        -1.0 + 5.0 * epsilon * pi**2 + reaction_alpha
    )

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Solution at current and previous time step
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step

    dt_const = fem.Constant(domain, ScalarType(dt))
    eps_const = fem.Constant(domain, ScalarType(epsilon))
    alpha_const = fem.Constant(domain, ScalarType(reaction_alpha))

    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + alpha*u = f
    # Weak form: (u - u_n)/dt * v + eps * grad(u) . grad(v) + alpha * u * v = f * v
    a = (u * v / dt_const + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v)) +
         alpha_const * u * v) * ufl.dx
    L_form = (u_n * v / dt_const + f_ufl * v) * ufl.dx

    # Boundary conditions - all boundary
    def boundary_all(x_arr):
        return (np.isclose(x_arr[0], 0.0) | np.isclose(x_arr[0], 1.0) |
                np.isclose(x_arr[1], 0.0) | np.isclose(x_arr[1], 1.0))

    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_all)
    bc_func = fem.Function(V)

    # Set initial BC values at t=0
    def u_exact_func(x_arr, t_val):
        return np.exp(-t_val) * np.cos(2 * np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1])

    bc_func.interpolate(lambda x_arr: u_exact_func(x_arr, 0.0))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(bc_func, dofs)

    # Set initial condition
    u_n.interpolate(lambda x_arr: u_exact_func(x_arr, 0.0))

    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 65, 65
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Compile forms
    a_compiled = fem.form(a)
    L_compiled = fem.form(L_form)

    # Assemble matrix (constant in time for linear reaction)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # Create RHS vector
    b = fem.petsc.create_vector(L_compiled)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    # Time stepping
    n_steps = int(np.round(t_end / dt))
    t_current = 0.0
    total_iterations = 0

    for step in range(n_steps):
        t_current += dt
        t_const.value = t_current

        # Update boundary condition
        bc_func.interpolate(lambda x_arr, t=t_current: u_exact_func(x_arr, t))

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate solution on output grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition
    u_n_init = fem.Function(V)
    u_n_init.interpolate(lambda x_arr: u_exact_func(x_arr, 0.0))

    u_init_values = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n_init.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()

    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }