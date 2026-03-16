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
    N = 64
    degree = 2
    dt_val = 0.005
    t_end = 0.1
    n_steps = int(round(t_end / dt_val))

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))

    # Manufactured solution: u = exp(-t)*sin(2*pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])

    # kappa = 1 + 0.5*sin(6*pi*x)
    kappa = 1.0 + 0.5 * ufl.sin(6 * pi * x[0])

    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(pi*y)
    du_dt = -ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    f = du_dt + ufl.div(-kappa * grad_u_exact)

    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler: (u - u_n)/dt - div(kappa * grad(u)) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    a = (u_h * v / dt) * ufl.dx + kappa * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt) * ufl.dx + f * v * ufl.dx

    # Boundary conditions: u = u_exact on all boundaries
    u_bc = fem.Function(V)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Expression for exact solution interpolation
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # Set initial condition at t=0
    t.value = 0.0
    u_n.interpolate(u_exact_expr)

    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    u_sol = fem.Function(V)

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val

        # Update BC
        u_bc.interpolate(u_exact_expr)
        bc = fem.dirichletbc(u_bc, dofs)

        # Assemble
        A = petsc.assemble_matrix(a_form, bcs=[bc])
        A.assemble()

        b = petsc.create_vector(L_form)
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.setOperators(A)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]

        A.destroy()
        b.destroy()

    # Evaluate solution on grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_2d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(points_2d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition for output
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_exact_expr)

    u_init_values = np.full(points_2d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()

    u_initial_grid = u_init_values.reshape((nx_out, ny_out))

    solver.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }