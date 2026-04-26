import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})

    t0 = time_info.get("t0", 0.0)
    t_end = time_info.get("t_end", 0.4)
    dt_suggested = time_info.get("dt", 0.02)
    scheme = time_info.get("scheme", "backward_euler")

    epsilon = pde.get("epsilon", pde.get("eps", 1.0))
    alpha = pde.get("reaction_coeff", pde.get("alpha", pde.get("reaction_rate", 1.0)))

    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    N = 64
    degree = 2
    dt = dt_suggested
    rtol = 1e-10

    comm = MPI.COMM_WORLD

    xmin, xmax, ymin, ymax = bbox
    p0 = np.array([xmin, ymin], dtype=np.float64)
    p1 = np.array([xmax, ymax], dtype=np.float64)
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N],
                                   cell_type=mesh.CellType.triangle)

    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    pi = np.pi

    # BC: u = 0 on boundary
    boundary_facets = mesh.locate_entities_boundary(domain, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # Time as fem.Constant
    t_const = fem.Constant(domain, ScalarType(t0))

    x = ufl.SpatialCoordinate(domain)
    sin_part = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Source: f = (-1 + 2*eps*pi^2 + alpha) * exp(-t) * sin(pi*x)*sin(pi*y)
    f_coeff = -1.0 + 2.0 * epsilon * pi**2 + alpha
    f_ufl = f_coeff * ufl.exp(-t_const) * sin_part

    u_n = fem.Function(V)
    u_sol = fem.Function(V)

    # Initial condition
    u_n.interpolate(lambda x: np.sin(pi * x[0]) * np.sin(pi * x[1]))

    # Backward Euler form
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_ufl = (1.0/dt) * ufl.inner(u_trial, v) * ufl.dx \
            + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx \
            + alpha * ufl.inner(u_trial, v) * ufl.dx

    L_ufl = ufl.inner(f_ufl, v) * ufl.dx \
            + (1.0/dt) * ufl.inner(u_n, v) * ufl.dx

    a_form = fem.form(a_ufl)
    L_form = fem.form(L_ufl)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")
    ksp.setTolerances(rtol=rtol)
    ksp.setFromOptions()

    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0

    for step in range(n_steps):
        t_current = t0 + (step + 1) * dt
        t_const.value = ScalarType(t_current)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]

    # L2 error verification
    u_exact_ufl = ufl.exp(-t_end) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    error_form = fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
    l2_error = np.sqrt(error_global) if domain.comm.rank == 0 else 0.0
    l2_error = domain.comm.bcast(l2_error, root=0)
    if domain.comm.rank == 0:
        print(f"L2 error at t={t_end}: {l2_error:.6e}")

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny_out, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    if domain.comm.size > 1:
        u_global = np.zeros_like(u_values)
        domain.comm.Allreduce(u_values, u_global, op=MPI.SUM)
        u_values = u_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # Initial condition on grid
    u_init_func = fem.Function(V)
    u_init_func.interpolate(lambda x: np.sin(pi * x[0]) * np.sin(pi * x[1]))
    u_init_vals = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vi = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_vals[eval_map] = vi.flatten()

    if domain.comm.size > 1:
        ui_global = np.zeros_like(u_init_vals)
        domain.comm.Allreduce(u_init_vals, ui_global, op=MPI.SUM)
        u_init_vals = ui_global

    u_initial_grid = u_init_vals.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
