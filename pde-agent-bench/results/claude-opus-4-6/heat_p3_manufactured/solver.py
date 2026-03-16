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
    kappa = 1.0
    t_end = 0.08
    dt_suggested = 0.008
    mesh_resolution = 64
    element_degree = 3

    # Use smaller dt for accuracy with higher-order elements
    dt = dt_suggested / 2.0  # 0.004
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # ensure exact final time

    # Create mesh
    nx = ny = mesh_resolution
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Time and spatial coordinates
    t = fem.Constant(domain, ScalarType(0.0))
    x = ufl.SpatialCoordinate(domain)

    # Exact solution: u = exp(-t)*sin(pi*x)*sin(2*pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(pi*x)*sin(2*pi*y)
    # laplacian(u) = exp(-t)*(-pi^2 - 4*pi^2)*sin(pi*x)*sin(2*pi*y) = -5*pi^2*exp(-t)*sin(pi*x)*sin(2*pi*y)
    # f = -exp(-t)*sin(pi*x)*sin(2*pi*y) - kappa*(-5*pi^2*exp(-t)*sin(pi*x)*sin(2*pi*y))
    # f = exp(-t)*sin(pi*x)*sin(2*pi*y)*(-1 + 5*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1]) * (-1.0 + 5.0 * kappa * pi**2)

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step

    # Initial condition
    t.value = 0.0
    u_exact_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_exact_expr)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

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
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for this problem)
    # But BCs change with time, so we need to reassemble with BCs each step
    # Actually, the matrix entries don't change, only the BC values change
    # For simplicity, assemble once and update BCs

    # Setup solver
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, dofs)])
    A.assemble()

    b = fem.petsc.create_vector(L_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iterations = 0

    # Store initial condition for output
    # Evaluate on grid for u_initial
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Build point evaluation infrastructure
    bb_tree = geometry.bb_tree(domain, tdim)
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

    def evaluate_function(func):
        values = np.full(points_3d.shape[1], np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            values[eval_map] = vals.flatten()
        return values.reshape(nx_out, ny_out)

    u_initial = evaluate_function(u_n)

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt

        # Update BC
        u_bc.interpolate(u_exact_expr)
        bc = fem.dirichletbc(u_bc, dofs)

        # Reassemble matrix with updated BCs (matrix is same but BC rows change)
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
    u_grid = evaluate_function(u_h)

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }