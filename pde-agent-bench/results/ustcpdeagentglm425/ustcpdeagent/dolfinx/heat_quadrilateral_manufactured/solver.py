import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    pde = case_spec["pde"]
    coeff = pde.get("coefficients", {})
    kappa = coeff.get("kappa", 1.0) if isinstance(coeff, dict) else 1.0
    time_info = pde.get("time", {})
    t0 = time_info.get("t0", 0.0)
    t_end = time_info.get("t_end", 0.1)
    dt_suggested = time_info.get("dt", 0.01)

    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    # Solver parameters
    mesh_res = 64
    elem_degree = 2
    dt = 0.005

    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps

    # Create quadrilateral mesh
    comm = MPI.COMM_WORLD
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(comm, [p0, p1], [mesh_res, mesh_res],
                                   cell_type=mesh.CellType.quadrilateral)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Functions for time stepping
    u_n = fem.Function(V)
    u_next = fem.Function(V)
    f_func = fem.Function(V)

    # Initial condition: u_0 = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Boundary condition: homogeneous Dirichlet on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    # Variational form for backward Euler
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a = (ufl.inner(u_trial, v_test) * ufl.dx
         + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx)
    L = (ufl.inner(u_n, v_test) * ufl.dx
         + dt * ufl.inner(f_func, v_test) * ufl.dx)

    a_compiled = fem.form(a)
    L_compiled = fem.form(L)

    # Assemble matrix A
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    # Create RHS vector
    b = petsc.create_vector(L_compiled.function_spaces)

    # Setup KSP solver
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol, atol=1e-14, max_it=500)

    # Time stepping loop
    t = t0
    total_iterations = 0

    for step in range(n_steps):
        t_new = t + dt

        # Update source term f = (2*pi^2 - 1)*exp(-t_new)*sin(pi*x)*sin(pi*y)
        coeff_f = (2.0 * np.pi**2 - 1.0) * np.exp(-t_new)
        f_func.interpolate(
            lambda x, cf=coeff_f: cf * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)

        # Apply lifting for BCs
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        # Solve linear system
        ksp.solve(b, u_next.x.petsc_vec)
        u_next.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_next.x.array[:]
        t = t_new

    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()

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

    u_grid = np.full((nx_out * ny_out,), np.nan)
    pts_arr = None
    cells_arr = None
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_next.eval(pts_arr, cells_arr)
        u_grid[eval_map] = vals.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)

    # Sample initial condition
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    u_initial_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_initial_grid[eval_map] = vals_init.flatten()
    u_initial_grid = u_initial_grid.reshape(ny_out, nx_out)

    # Accuracy verification
    u_exact_final = np.exp(-t_end) * np.sin(np.pi * XX) * np.sin(np.pi * YY)
    error_grid = u_grid - u_exact_final
    valid = ~np.isnan(error_grid)
    if np.any(valid):
        max_error_grid = np.max(np.abs(error_grid[valid]))
        l2_error_grid = np.sqrt(np.mean(error_grid[valid]**2))
    else:
        max_error_grid = float('inf')
        l2_error_grid = float('inf')

    print(f"[Solver] Grid L2 error: {l2_error_grid:.6e}, Grid max error: {max_error_grid:.6e}")
    print(f"[Solver] Mesh res: {mesh_res}, Element degree: {elem_degree}, dt: {dt}, Steps: {n_steps}")
    print(f"[Solver] Total KSP iterations: {total_iterations}")

    # Explicitly destroy PETSc objects to avoid slow cleanup
    ksp.destroy()
    A.destroy()
    b.destroy()

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }
