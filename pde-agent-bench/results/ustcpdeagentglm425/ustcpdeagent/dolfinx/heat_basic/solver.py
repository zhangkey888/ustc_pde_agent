import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    pde = case_spec["pde"]
    coeff = pde.get("coefficients", {})
    kappa = coeff.get("kappa", 1.0)
    time_params = pde.get("time", {})
    t0 = time_params.get("t0", 0.0)
    t_end = time_params.get("t_end", 0.1)
    dt_default = time_params.get("dt", 0.01)

    # Output grid
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Choose solver parameters for accuracy
    mesh_res = 112
    element_degree = 2
    dt = 0.00125  # smaller dt for better temporal accuracy
    n_steps = int(round((t_end - t0) / dt))
    time_scheme = "backward_euler"
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    pi = np.pi

    # Current and next solution
    u_n = fem.Function(V)  # solution at previous time step
    u_h = fem.Function(V)  # solution at current time step (unknown)

    # Test function
    v = ufl.TestFunction(V)

    # Source term function (will be updated each step)
    f_func = fem.Function(V)

    # Dirichlet BC function (will be updated each step)
    u_bc_func = fem.Function(V)

    # Boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # Variational form for backward Euler:
    # (u^{n+1} - u^n)/dt - div(kappa * grad(u^{n+1})) = f^{n+1}
    u_trial = ufl.TrialFunction(V)
    a = ufl.inner(u_trial, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx

    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix (constant since mesh and dt don't change)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Create solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    # Initial condition: u(x,0) = sin(pi*x)*sin(pi*y)
    u_n.interpolate(lambda x: np.sin(pi * x[0]) * np.sin(pi * x[1]))

    # Point evaluation helper
    def evaluate_on_grid(u_func, nx_out, ny_out, bbox):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        points = np.zeros((3, nx_out * ny_out))
        points[0, :] = XX.ravel()
        points[1, :] = YY.ravel()
        points[2, :] = 0.0

        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
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

        u_values = np.full((points.shape[1],), np.nan)
        if len(points_on_proc) > 0:
            pts = np.array(points_on_proc)
            cells = np.array(cells_on_proc, dtype=np.int32)
            vals = u_func.eval(pts, cells)
            u_values[eval_map] = vals.flatten()
        return u_values.reshape(ny_out, nx_out)

    # Get initial condition on grid
    u_initial_grid = evaluate_on_grid(u_n, nx_out, ny_out, bbox)

    # Helper functions
    def set_exact_solution(func, t):
        func.interpolate(lambda x: np.exp(-t) * np.sin(pi * x[0]) * np.sin(pi * x[1]))

    def set_source(func, t):
        func.interpolate(lambda x: (2 * pi**2 - 1) * np.exp(-t) * np.sin(pi * x[0]) * np.sin(pi * x[1]))

    # Time stepping
    total_iterations = 0
    t = t0

    # RHS vector
    b = petsc.create_vector(L_form.function_spaces)

    for step in range(n_steps):
        t_next = t + dt

        # Update source term at t_{n+1}
        set_source(f_func, t_next)

        # Update boundary condition at t_{n+1}
        set_exact_solution(u_bc_func, t_next)

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

        # Count iterations
        total_iterations += solver.getIterationNumber()

        # Update: u_n <- u_h
        u_n.x.array[:] = u_h.x.array[:]
        t = t_next

    # Evaluate final solution on output grid
    u_grid = evaluate_on_grid(u_h, nx_out, ny_out, bbox)

    # Compute L2 error against exact solution
    u_exact_func = fem.Function(V)
    set_exact_solution(u_exact_func, t_end)
    error_sq = ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx
    error_form = fem.form(error_sq)
    L2_error_local = fem.assemble_scalar(error_form)
    L2_error = np.sqrt(np.abs(comm.allreduce(L2_error_local, op=MPI.SUM)))
    if comm.rank == 0:
        print(f"L2 error at t={t_end}: {L2_error:.6e}")
        print(f"n_steps={n_steps}, dt={dt}, mesh_res={mesh_res}, degree={element_degree}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
    }

    result = {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }

    return result
