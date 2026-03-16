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
    N = 80  # mesh resolution
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

    # Exact solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Diffusion coefficient: kappa = 1 + 30*exp(-150*((x-0.35)^2 + (y-0.65)^2))
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))

    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(pi*x)*sin(2*pi*y)
    dudt = -ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = dudt + ufl.div(-kappa * ufl.grad(u_exact_ufl))
    # Note: the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - div(kappa*grad(u_exact)) but div(kappa*grad(u)) appears with negative sign in PDE
    # Actually: du/dt - div(kappa * grad(u)) = f
    # => f = du/dt - div(kappa * grad(u_exact))
    # f = -exp(-t)*sin(pi*x)*sin(2*pi*y) - div(kappa * grad(exp(-t)*sin(pi*x)*sin(2*pi*y)))
    f_source = dudt - ufl.div(kappa * ufl.grad(u_exact_ufl))

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f
    # Weak form: (u - u_n)/dt * v dx + kappa*grad(u)·grad(v) dx = f*v dx
    a = (u * v / dt) * ufl.dx + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt) * ufl.dx + f_source * v * ufl.dx

    # Boundary conditions (Dirichlet, u = u_exact on boundary)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Set initial condition
    t.value = 0.0
    ic_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(ic_expr)

    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0] = XX.ravel()
    points_2d[1] = YY.ravel()

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (kappa doesn't depend on time, but a depends on dt which is constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Create RHS vector
    b = petsc.create_vector(L_form)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val

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
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]

    # Evaluate solution on output grid
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts, cls)
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition for output
    t.value = 0.0
    u_init_func = fem.Function(V)
    u_init_func.interpolate(ic_expr)

    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(pts, cls)
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
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }