import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict = None) -> dict:
    """Solve the transient heat equation using backward Euler."""

    if case_spec is None:
        case_spec = {}

    # ---- Parse case_spec (handle both direct and oracle_config nesting) ----
    oracle_config = case_spec.get("oracle_config", case_spec)
    pde = oracle_config.get("pde", case_spec.get("pde", {}))

    # Coefficients
    coeffs = pde.get("coefficients", {})
    kappa_spec = coeffs.get("kappa", {})
    if isinstance(kappa_spec, dict):
        kappa = float(kappa_spec.get("value", 1.0))
    else:
        kappa = float(kappa_spec)

    # Time parameters (with hardcoded defaults matching the problem)
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_val = float(time_params.get("dt", 0.01))  # Use smaller dt for better accuracy
    scheme = time_params.get("scheme", "backward_euler")

    # Output grid
    output = oracle_config.get("output", {})
    grid = output.get("grid", {})
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)

    # ---- Solver parameters ----
    N = 64
    degree = 1

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    # Initial condition
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))

    # Boundary conditions (homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]

    # Time stepping setup
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Backward Euler weak form
    a = (u * v / dt + kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v / dt + f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.create_vector(V)

    u_sol = fem.Function(V, name="u")

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()

    # Compute actual number of steps and adjusted dt
    n_steps = int(round(t_end / dt_val))
    dt_actual = t_end / n_steps
    if abs(dt_actual - dt_val) > 1e-14:
        dt.value = dt_actual
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()

    t = 0.0
    total_iterations = 0

    for step in range(n_steps):
        t += dt_actual

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]

    # ---- Evaluate on output grid ----
    x_grid = np.linspace(0, 1, nx_out)
    y_grid = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.flatten()
    points[1, :] = Y.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full(nx_out * ny_out, np.nan)
    u_init_values = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[eval_map] = vals_init.flatten()

    u_grid = u_values.reshape((nx_out, ny_out))
    u_init_grid = u_init_values.reshape((nx_out, ny_out))

    solver.destroy()
    A.destroy()
    b.destroy()

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt_actual,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time

    start = time.time()
    result = solve()
    elapsed = time.time() - start
    print(f"Solve time: {elapsed:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min: {np.nanmin(result['u']):.6f}, max: {np.nanmax(result['u']):.6f}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
