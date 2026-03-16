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
    epsilon = 0.1
    beta = [1.0, 0.5]
    t_end = 0.1
    dt_val = 0.02
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

    # Time as a constant
    t = fem.Constant(domain, ScalarType(0.0))
    dt = fem.Constant(domain, ScalarType(dt_val))
    eps_c = fem.Constant(domain, ScalarType(epsilon))
    beta_vec = ufl.as_vector([fem.Constant(domain, ScalarType(beta[0])),
                               fem.Constant(domain, ScalarType(beta[1]))])

    # Exact solution: u = exp(-t)*sin(pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact_ufl = ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source term f = du/dt - eps*laplacian(u) + beta.grad(u)
    # du/dt = -exp(-t)*sin(pi*x)*sin(pi*y)
    # laplacian(u) = -2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y)
    # grad(u) = exp(-t)*[pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y)]
    # f = -exp(-t)*sin(pi*x)*sin(pi*y) - eps*(-2*pi^2*exp(-t)*sin(pi*x)*sin(pi*y))
    #     + beta[0]*exp(-t)*pi*cos(pi*x)*sin(pi*y) + beta[1]*exp(-t)*pi*sin(pi*x)*cos(pi*y)
    f_expr = (
        -ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        + eps_c * 2.0 * pi**2 * ufl.exp(-t) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
        + beta_vec[0] * ufl.exp(-t) * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
        + beta_vec[1] * ufl.exp(-t) * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    )

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")

    # Interpolate initial condition at t=0
    u_n.interpolate(lambda x_arr: np.sin(np.pi * x_arr[0]) * np.sin(np.pi * x_arr[1]))

    # Store initial condition for output
    # Build evaluation grid first
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])

    # Evaluate initial condition
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

    u_initial_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals.flatten()
    u_initial = u_initial_vals.reshape((nx_out, ny_out))

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_cell = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form: (u - u_n)/dt * v + eps*grad(u).grad(v) + beta.grad(u)*v = f*v
    # With SUPG: add stabilization term

    # Bilinear form
    a_form = (
        u / dt * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
        # SUPG stabilization on LHS
        + tau * (u / dt - eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_vec, ufl.grad(u))) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    )

    # For SUPG with trial functions, the second-order term div(grad(u)) on trial function
    # For Lagrange degree >= 2, this is not zero element-wise but can cause issues.
    # Simpler: drop the diffusion part in SUPG residual (common approximation)
    # Let's use a cleaner SUPG formulation
    a_form = (
        u / dt * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta_vec, ufl.grad(u)) * v * ufl.dx
        # SUPG
        + tau * (u / dt + ufl.dot(beta_vec, ufl.grad(u))) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    )

    L_form = (
        u_n / dt * v * ufl.dx
        + f_expr * v * ufl.dx
        # SUPG on RHS
        + tau * (u_n / dt + f_expr) * ufl.dot(beta_vec, ufl.grad(v)) * ufl.dx
    )

    # Boundary conditions: u = g = exact solution on boundary
    u_bc = fem.Function(V)

    # Mark all boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x_arr: np.ones(x_arr.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix (changes each step due to dt being constant, but actually dt doesn't change)
    # Matrix doesn't change between steps since dt and coefficients are constant
    # But BC values change with time, so we need to reassemble or use lifting properly

    # Solution function
    u_sol = fem.Function(V, name="u_sol")

    # Setup solver
    A = petsc.assemble_matrix(a_compiled, bcs=[fem.dirichletbc(u_bc, dofs)])
    A.assemble()

    b = petsc.create_vector(L_compiled)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    total_iterations = 0

    # Create expression for BC interpolation
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt_val

        # Update BC
        u_bc.interpolate(bc_expr)
        bc = fem.dirichletbc(u_bc, dofs)

        # Reassemble matrix (BCs might affect pattern - safer to reassemble)
        A.zeroEntries()
        petsc.assemble_matrix(A, a_compiled, bcs=[bc])
        A.assemble()

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]

    # Evaluate solution on grid
    u_grid_vals = np.full(points_3d.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_vals[eval_map] = vals.flatten()
    u_grid = u_grid_vals.reshape((nx_out, ny_out))

    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }