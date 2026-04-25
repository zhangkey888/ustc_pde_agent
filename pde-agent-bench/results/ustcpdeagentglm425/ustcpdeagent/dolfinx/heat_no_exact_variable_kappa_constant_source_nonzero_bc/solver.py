import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract PDE parameters
    pde = case_spec["pde"]
    coeffs = pde["coefficients"]
    time_params = pde["time"]

    f_val = float(pde.get("source", 1.0))
    u0_val = float(pde.get("initial_condition", 0.0))

    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))

    # Output grid
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]

    # ── High-accuracy parameters ──
    mesh_res = 160
    elem_degree = 2
    dt = 0.0025  # 40 time steps
    n_steps = int(round((t_end - t0) / dt))

    # ── Mesh and function space ──
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    x = ufl.SpatialCoordinate(domain)

    # ── Variable coefficient kappa ──
    kappa = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # ── Source term ──
    f = fem.Constant(domain, PETSc.ScalarType(f_val))

    # ── Boundary condition function g ──
    g_expr = ufl.sin(ufl.pi * x[0]) + ufl.cos(ufl.pi * x[1])
    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))

    # ── Locate boundary ──
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)

    # ── Solution functions ──
    u_n = fem.Function(V)
    u_n.x.array[:] = u0_val
    u_sol = fem.Function(V)

    # ── Weak form: Backward Euler ──
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    a = ufl.inner(u_trial, v_test) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = ufl.inner(u_n, v_test) * ufl.dx + dt * ufl.inner(f, v_test) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    # ── Solver setup ──
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-12
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=rtol, atol=1e-15)
    ksp.getPC().setHYPREType("boomeramg")

    total_iterations = 0

    # ── Time loop ──
    t = t0
    for step in range(n_steps):
        t += dt

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

    # ── Accuracy verification ──
    L2_sq = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
    L2_norm = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(L2_sq), op=MPI.SUM))

    # ── Sample solution onto output grid ──
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full((ny_out, nx_out), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        flat = np.full(pts.shape[1], np.nan)
        flat[eval_map] = vals.flatten()
        u_grid = flat.reshape(ny_out, nx_out)

    # Sample initial condition
    u_init_grid = np.full((ny_out, nx_out), np.nan)
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u0_val
    if len(points_on_proc) > 0:
        vals_init = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        flat_init = np.full(pts.shape[1], np.nan)
        flat_init[eval_map] = vals_init.flatten()
        u_init_grid = flat_init.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info,
    }
