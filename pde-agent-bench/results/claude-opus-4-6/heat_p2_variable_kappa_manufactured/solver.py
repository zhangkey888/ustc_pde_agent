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
    t_end = 0.06
    n_steps = int(round(t_end / dt_val))

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))

    # Manufactured solution: u_exact = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])

    # Variable kappa: 1 + 0.4*sin(2*pi*x)*sin(2*pi*y)
    kappa = 1.0 + 0.4 * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])

    # Source term: f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # We compute f symbolically
    du_dt = -ufl.exp(-t) * ufl.sin(2 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    grad_u_exact = ufl.grad(u_exact_ufl)
    div_kappa_grad_u = ufl.div(kappa * grad_u_exact)
    f = du_dt - div_kappa_grad_u

    # Functions
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Initial condition
    t.value = 0.0
    u_init_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    # Store initial condition for output
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()

    # Prepare point evaluation infrastructure
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

    points_on_proc_arr = np.array(points_on_proc) if len(points_on_proc) > 0 else np.empty((0, 3))
    cells_on_proc_arr = np.array(cells_on_proc, dtype=np.int32) if len(cells_on_proc) > 0 else np.empty(0, dtype=np.int32)

    def evaluate_function(func):
        values = np.full(nx_out * ny_out, np.nan)
        if len(points_on_proc) > 0:
            vals = func.eval(points_on_proc_arr, cells_on_proc_arr)
            values[eval_map] = vals.flatten()
        return values.reshape((nx_out, ny_out))

    u_initial = evaluate_function(u_n)

    # Backward Euler: (u - u_n)/dt - div(kappa*grad(u)) = f
    # Bilinear form: (u, v)/dt + (kappa*grad(u), grad(v)) = (u_n, v)/dt + (f, v)
    a = (u_h * v / dt_c) * ufl.dx + ufl.inner(kappa * ufl.grad(u_h), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f * v * ufl.dx

    # Boundary conditions (Dirichlet, from exact solution)
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (kappa doesn't depend on time, and a doesn't change)
    A = petsc.assemble_matrix(a_form, bcs=[fem.dirichletbc(u_bc, bc_dofs)])
    A.assemble()

    b = petsc.create_vector(L_form)

    # Setup KSP solver
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
        bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
        u_bc.interpolate(bc_expr)
        bc = fem.dirichletbc(u_bc, bc_dofs)

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
    u_grid = evaluate_function(u_sol)

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial,
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