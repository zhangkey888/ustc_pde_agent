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
    pde = case_spec["pde"]
    eps_val = pde["coefficients"]["epsilon"]
    beta_val = pde["coefficients"]["beta"]
    time_params = pde["time"]
    t_end = time_params["t_end"]
    dt_suggested = time_params["dt"]

    # Choose parameters
    N = 80  # mesh resolution
    degree = 1
    dt = dt_suggested  # 0.02
    n_steps = int(round(t_end / dt))

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Time constant
    t = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))

    # Coefficients
    epsilon = fem.Constant(domain, ScalarType(eps_val))
    beta = fem.Constant(domain, np.array(beta_val, dtype=ScalarType))

    # Source term: f = exp(-150*((x-0.4)**2 + (y-0.6)**2))*exp(-t)
    f_expr = ufl.exp(-150.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2)) * ufl.exp(-t)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Previous solution
    u_n = fem.Function(V, name="u_n")

    # Initial condition: u0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    tau = h / (2.0 * beta_norm) * (ufl.conditional(ufl.gt(Pe_cell, 1.0),
                                                      1.0 - 1.0 / Pe_cell,
                                                      0.0))

    # Backward Euler: (u - u_n)/dt - eps*laplacian(u) + beta.grad(u) = f
    # Weak form (Galerkin part):
    # (u - u_n)/dt * v + eps * grad(u).grad(v) + (beta.grad(u)) * v = f * v
    a_galerkin = (u / dt_const * v
                  + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v))
                  + ufl.dot(beta, ufl.grad(u)) * v) * ufl.dx

    L_galerkin = (u_n / dt_const * v + f_expr * v) * ufl.dx

    # SUPG stabilization terms
    # Residual of the strong form applied to trial function (linearized):
    # R = u/dt - eps*laplacian(u) + beta.grad(u) - f
    # For linear elements, laplacian(u) = 0 within each cell
    # So R_trial = u/dt + beta.grad(u)
    # R_rhs = u_n/dt + f

    # SUPG test function modification: v_supg = tau * beta.grad(v)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    a_supg = (u / dt_const * v_supg
              + ufl.dot(beta, ufl.grad(u)) * v_supg) * ufl.dx

    L_supg = (u_n / dt_const * v_supg + f_expr * v_supg) * ufl.dx

    a = a_galerkin + a_supg
    L = L_galerkin + L_supg

    # Boundary conditions: u = 0 on all boundaries (g = 0 for homogeneous)
    # The problem says u = g on boundary. Since no explicit g is given, 
    # and sin(pi*x)*sin(pi*y) = 0 on boundary, use homogeneous BC
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant in time since coefficients don't change with time for LHS)
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(L_form)

    # Solution function
    u_sol = fem.Function(V, name="u")

    # Setup KSP solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    solver.setUp()

    total_iterations = 0

    # Time stepping
    for step in range(n_steps):
        t.value = (step + 1) * dt

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

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

    u_grid = np.full(nx_out * ny_out, np.nan)
    u_init_grid = np.full(nx_out * ny_out, np.nan)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_grid[eval_map] = vals_init.flatten()

    u_grid = u_grid.reshape((nx_out, ny_out))
    u_init_grid = u_init_grid.reshape((nx_out, ny_out))

    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }