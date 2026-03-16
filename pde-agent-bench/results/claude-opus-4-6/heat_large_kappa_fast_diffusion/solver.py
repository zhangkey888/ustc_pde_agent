import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    kappa = 5.0
    t_end = 0.08
    dt = 0.004
    n_steps = int(round(t_end / dt))
    mesh_res = 80
    degree = 2

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Time as a constant
    t = fem.Constant(domain, ScalarType(0.0))

    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(pi*y)
    # laplacian(u) = exp(-t) * (-(2*pi)^2 - pi^2) * sin(2*pi*x)*sin(pi*y)
    #              = -5*pi^2 * exp(-t)*sin(2*pi*x)*sin(pi*y)
    # f = du/dt - kappa * laplacian(u)
    #   = -exp(-t)*sin(2*pi*x)*sin(pi*y) - kappa*(-5*pi^2)*exp(-t)*sin(2*pi*x)*sin(pi*y)
    #   = exp(-t)*sin(2*pi*x)*sin(pi*y) * (-1 + 5*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 + 5.0 * kappa * ufl.pi**2)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)

    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))

    a = (u * v / dt_const + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_const + f_ufl * v) * ufl.dx

    # Boundary conditions - all boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    # Set initial BC
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Initial condition
    t.value = 0.0
    u_n.interpolate(bc_expr)

    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for this problem)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Create RHS vector
    b = fem.petsc.create_vector(L_form)

    # Solution function
    u_sol = fem.Function(V)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt

        # Update BC
        u_bc.interpolate(bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
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

    # Evaluate solution on grid
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)

    n_points = points_3d.shape[1]
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(bc_expr)

    u_init_values = np.full(n_points, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
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
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }