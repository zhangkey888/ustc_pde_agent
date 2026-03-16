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
    kappa = case_spec["pde"]["coefficients"]["kappa"]
    t_end = case_spec["pde"]["time"]["t_end"]
    dt_suggested = case_spec["pde"]["time"]["dt"]
    scheme = case_spec["pde"]["time"]["scheme"]

    # High-frequency in x (8*pi*x) needs fine mesh in x direction
    # For sin(8*pi*x), we need at least ~16 elements per wavelength
    # wavelength = 1/8, so need ~16/8 = 128 elements minimum, use more for accuracy
    # With P2 elements we can use fewer cells
    nx = 128
    ny = 32
    degree = 2

    # Use smaller dt for accuracy with high frequency
    dt = dt_suggested / 2.0  # 0.002
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # adjust to hit t_end exactly

    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Time and spatial coordinates
    t = fem.Constant(domain, ScalarType(0.0))
    x = ufl.SpatialCoordinate(domain)

    # Exact solution: u = exp(-t)*sin(8*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(8*pi*x)*sin(pi*y)
    # laplacian(u) = exp(-t)*(-64*pi^2 - pi^2)*sin(8*pi*x)*sin(pi*y)
    # f = -exp(-t)*sin(8*pi*x)*sin(pi*y) - kappa*exp(-t)*(-65*pi^2)*sin(8*pi*x)*sin(pi*y)
    # f = exp(-t)*sin(8*pi*x)*sin(pi*y)*(-1 + 65*kappa*pi^2)
    f_ufl = ufl.exp(-t) * ufl.sin(8 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * (-1.0 + 65.0 * kappa * ufl.pi**2)

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step

    # Initial condition
    t.value = 0.0
    u_n_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_n_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Boundary conditions - all boundaries are Dirichlet
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Backward Euler: (u - u_n)/dt - kappa * laplacian(u) = f(t_{n+1})
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))

    a = (u * v + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, bc_dofs)])
    A.assemble()

    b = fem.petsc.create_vector(L_form)

    # Setup solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setUp()

    total_iterations = 0

    for step in range(n_steps):
        t.value = (step + 1) * dt

        # Update BC
        u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(u_bc_expr)
        bc = fem.dirichletbc(u_bc, bc_dofs)

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

    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = X.ravel()
    points[1] = Y.ravel()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))

    # Also evaluate initial condition
    u_init_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_init_grid = u_init_values.reshape((nx_out, ny_out))

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
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
    }