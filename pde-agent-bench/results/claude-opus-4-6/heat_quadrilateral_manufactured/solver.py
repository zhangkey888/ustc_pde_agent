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
    dt = 0.01
    n_steps = int(round(t_end / dt))
    mesh_resolution = 64
    element_degree = 2

    # Create mesh (quadrilateral)
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.quadrilateral
    )
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Time variable
    t = fem.Constant(domain, ScalarType(0.0))

    # Manufactured solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # -kappa * laplacian(u) = kappa * 2*pi^2 * exp(-t)*sin(pi*x)*sin(pi*y)
    # f = du/dt - kappa*laplacian(u) ... wait, the PDE is du/dt - div(kappa grad u) = f
    # So f = du/dt - kappa * laplacian(u)
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) + kappa*2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # f = exp(-t)*sin(pi*x)*sin(pi*y)*(-1 + 2*kappa*pi^2)

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * (-1.0 + 2.0 * kappa * pi**2)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)

    # Initial condition
    t.value = 0.0
    u_n.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]

    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u - u_n)/dt * v dx + kappa * grad(u) . grad(v) dx = f * v dx
    # => u*v dx + dt*kappa*grad(u).grad(v) dx = u_n*v dx + dt*f*v dx

    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))

    a = (u * v + dt_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_const * f_expr * v) * ufl.dx

    # Boundary conditions (all boundary)
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for this problem)
    # Actually, BCs change with time, but the matrix entries on interior DOFs don't change.
    # We need to reassemble with BCs each step since BC values change.

    # Solution function
    u_sol = fem.Function(V)

    # Setup solver
    A = petsc.assemble_matrix(a_form, bcs=[])
    A.assemble()
    b = petsc.create_vector(L_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iterations = 0

    for step in range(n_steps):
        t.value = (step + 1) * dt

        # Update BC
        u_bc.interpolate(
            fem.Expression(u_exact_ufl, V.element.interpolation_points)
        )
        bc = fem.dirichletbc(u_bc, dofs)

        # Reassemble matrix with new BCs
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        solver.setOperators(A)

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
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

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