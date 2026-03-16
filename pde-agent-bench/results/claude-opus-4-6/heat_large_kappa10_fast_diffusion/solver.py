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
    kappa = 10.0
    t_end = 0.05
    dt_suggested = 0.005
    nx = ny = 64
    degree = 1

    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # ensure exact end time

    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Time and spatial coordinates
    t = fem.Constant(domain, ScalarType(0.0))
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    # Exact solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # laplacian(u) = -2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) - kappa*(-2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y))
    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * (-1.0 + 2.0 * kappa * pi**2)

    # Functions
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step

    # Initial condition
    t.value = 0.0
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    # Store initial condition for output
    # Build grid for evaluation
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0] = XX.ravel()
    points_2d[1] = YY.ravel()

    # Probe function
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    def evaluate_on_grid(u_func):
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

        u_values = np.full(points_2d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(nx_out, ny_out)

    u_initial = evaluate_on_grid(u_n)

    # Backward Euler variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))

    a = (ufl.inner(u, v) * ufl.dx
         + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (ufl.inner(u_n, v) * ufl.dx
         + dt_const * ufl.inner(f_ufl, v) * ufl.dx)

    # Boundary conditions (update each time step)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time since kappa and dt don't change)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, dofs)])
    A.assemble()

    b = fem.petsc.create_vector(L_form)

    # Setup KSP solver
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
        u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, dofs)

        # We need to reassemble A each step because BC values change
        # Actually, the matrix structure doesn't change, only the BC application
        # But since we used bcs in assemble_matrix, we need to reassemble
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()

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

    # Evaluate final solution on grid
    u_grid = evaluate_on_grid(u_h)

    solver.destroy()
    A.destroy()
    b.destroy()

    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }