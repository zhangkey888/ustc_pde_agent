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
    t_end = 0.1
    dt_val = 0.005
    n_steps = int(round(t_end / dt_val))
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

    # Time constant
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    # Exact solution: u = exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y))
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(3 * pi * (x[0] + x[1])) * ufl.sin(pi * (x[0] - x[1]))

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y)) = -u_exact
    # We need laplacian of u_exact w.r.t. spatial variables
    # Let's compute it symbolically via UFL
    grad_u_exact = ufl.grad(u_exact_ufl)
    laplacian_u_exact = ufl.div(grad_u_exact)
    f_ufl = -u_exact_ufl - kappa_c * laplacian_u_exact  # du/dt - kappa*laplacian = f => f = du/dt - kappa*laplacian

    # Wait: the PDE is du/dt - div(kappa*grad(u)) = f
    # So f = du/dt - kappa*laplacian(u)
    # du/dt of exact = -exp(-t)*sin(3*pi*(x+y))*sin(pi*(x-y)) = -u_exact
    # So f = -u_exact - kappa * laplacian(u_exact)
    # But laplacian(u_exact) = div(grad(u_exact))
    # f_ufl is correct as defined above

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Initial condition
    u_n_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    t.value = 0.0
    u_n.interpolate(u_n_expr)

    # Store initial condition for output
    # Evaluate on grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()

    # Build bounding box tree for point evaluation
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

    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)

    def evaluate_on_grid(u_func):
        u_values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(points_on_proc_arr, cells_on_proc_arr)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape((nx_out, ny_out))

    u_initial = evaluate_on_grid(u_n)

    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f(t_{n+1})
    # Weak form: (u_h - u_n)/dt * v dx + kappa * grad(u_h) . grad(v) dx = f * v dx
    a = (u_h * v / dt_c) * ufl.dx + kappa_c * ufl.inner(ufl.grad(u_h), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_ufl * v * ufl.dx

    # Boundary conditions - all boundary
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))

    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for this problem since kappa is constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form)

    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()

    u_sol = fem.Function(V)

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
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate final solution on grid
    u_grid = evaluate_on_grid(u_sol)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }