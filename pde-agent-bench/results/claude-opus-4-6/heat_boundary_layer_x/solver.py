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
    kappa = case_spec["pde"]["coefficients"]["kappa"]
    t_end = case_spec["pde"]["time"]["t_end"]
    dt_suggested = case_spec["pde"]["time"]["dt"]
    scheme = case_spec["pde"]["time"]["scheme"]

    # The manufactured solution has exp(5*x) which creates a boundary layer near x=1.
    # We need sufficient resolution. Use higher degree elements and refined mesh.
    nx = 80
    ny = 40
    degree = 2
    dt = dt_suggested  # 0.008
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust to hit t_end exactly

    # Create mesh
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    kappa_c = fem.Constant(domain, ScalarType(kappa))
    dt_c = fem.Constant(domain, ScalarType(dt))

    # Manufactured solution: u = exp(-t)*exp(5*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.exp(5.0 * x[0]) * ufl.sin(pi * x[1])

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*exp(5*x)*sin(pi*y)
    # laplacian(u) = exp(-t)*(25*exp(5*x)*sin(pi*y) - pi^2*exp(5*x)*sin(pi*y))
    #              = exp(-t)*exp(5*x)*sin(pi*y)*(25 - pi^2)
    # f = -u - kappa*(25 - pi^2)*u = u*(-1 - kappa*(25 - pi^2))
    f_ufl = -ufl.exp(-t) * ufl.exp(5.0 * x[0]) * ufl.sin(pi * x[1]) \
            - kappa_c * ufl.exp(-t) * (25.0 * ufl.exp(5.0 * x[0]) * ufl.sin(pi * x[1])
                                        - pi**2 * ufl.exp(5.0 * x[0]) * ufl.sin(pi * x[1]))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")

    # Interpolate initial condition at t=0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    t.value = 0.0
    u_n.interpolate(u_init_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u, v)/dt + kappa*(grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = (u * v / dt_c + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt_c + f_ufl * v) * ufl.dx

    # Boundary conditions - all boundary
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for backward Euler with constant kappa)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
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

    u_sol = fem.Function(V)

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

    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

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

    # Cleanup
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
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }