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

    # Spatial coordinate and time
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt_c = fem.Constant(domain, ScalarType(dt_val))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    # Exact solution as UFL expression
    u_exact_ufl = ufl.exp(-t) * ufl.exp(-40.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

    # Source term: f = du/dt - kappa * laplacian(u)
    # du/dt = -exp(-t)*exp(-40*((x-0.5)^2+(y-0.5)^2))
    # laplacian(u) = exp(-t) * exp(-40*r2) * (-160 + 6400*r2) where r2 = (x-0.5)^2+(y-0.5)^2
    # So f = -u - kappa * laplacian(u)
    # Let's compute it symbolically
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    gauss = ufl.exp(-40.0 * r2)
    u_exact = ufl.exp(-t) * gauss

    # du/dt = -exp(-t)*gauss = -u_exact
    dudt = -u_exact

    # laplacian of u: div(grad(u))
    # grad(u) = exp(-t) * exp(-40*r2) * [-80*(x-0.5), -80*(y-0.5)]
    # div(grad(u)) = exp(-t) * [(-80)*exp(-40*r2) + (-80*(x-0.5))*(-80*(x-0.5))*exp(-40*r2)
    #                           + (-80)*exp(-40*r2) + (-80*(y-0.5))*(-80*(y-0.5))*exp(-40*r2)]
    # = exp(-t)*exp(-40*r2)*[-160 + 6400*r2]
    lap_u = ufl.exp(-t) * gauss * (-160.0 + 6400.0 * r2)

    f_expr = dudt - kappa_c * lap_u

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V)

    # Interpolate initial condition
    u_n.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))

    # Store initial condition for output
    # Create grid for evaluation
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()

    # Evaluate initial condition
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

    u_initial_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals.flatten()
    u_initial = u_initial_vals.reshape((nx_out, ny_out))

    # Backward Euler: (u - u_n)/dt - kappa*laplacian(u) = f
    # Weak form: (u/dt)*v*dx + kappa*grad(u)*grad(v)*dx = (u_n/dt)*v*dx + f*v*dx
    a = (u / dt_c) * v * ufl.dx + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n / dt_c) * v * ufl.dx + f_expr * v * ufl.dx

    # Boundary conditions - use exact solution
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    # Will update BC each time step
    bc = fem.dirichletbc(u_bc, dofs)

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time for this problem since kappa is constant)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Create RHS vector
    b = petsc.create_vector(L_form)

    # Solution function
    u_sol = fem.Function(V)

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
    current_t = 0.0
    for step in range(n_steps):
        current_t += dt_val
        t.value = current_t

        # Update boundary condition
        u_bc.interpolate(
            fem.Expression(u_exact_ufl, V.element.interpolation_points)
        )

        # Reassemble matrix (BCs might change pattern - but here A is constant structure)
        # Actually for changing Dirichlet values, we need to reassemble with new BCs
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
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate final solution on grid
    u_final_vals = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_final_vals[eval_map] = vals.flatten()
    u_grid = u_final_vals.reshape((nx_out, ny_out))

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
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
    }