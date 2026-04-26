import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_ufl(msh, t):
    x = ufl.SpatialCoordinate(msh)
    return ufl.exp(-t) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])


def _kappa_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    return 1.0 + 0.3 * ufl.cos(2.0 * ufl.pi * x[0]) * ufl.cos(2.0 * ufl.pi * x[1])


def _sample_function(u_fun, grid_spec):
    msh = u_fun.function_space.mesh
    comm = msh.comm

    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idxs = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idxs.append(i)

    if points_on_proc:
        vals = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        local_vals[np.array(idxs, dtype=np.int32)] = np.asarray(vals).reshape(-1).real

    send = np.where(np.isnan(local_vals), 0.0, local_vals)
    recv = np.empty_like(send) if comm.rank == 0 else None
    comm.Reduce(send, recv, op=MPI.SUM, root=0)

    if comm.rank == 0:
        return recv.reshape((ny, nx))
    return None


def _compute_l2_error(u_h, u_ex):
    msh = u_h.function_space.mesh
    local = fem.assemble_scalar(fem.form((u_h - u_ex) ** 2 * ufl.dx))
    global_val = msh.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(max(global_val, 0.0))


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    time_spec = case_spec.get("pde", {}).get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))

    dt = min(dt_suggested, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    mesh_resolution = 48
    element_degree = 2

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Function(V)
    kappa.interpolate(fem.Expression(_kappa_ufl(msh), V.element.interpolation_points))

    u_n = fem.Function(V)
    u0_expr = fem.Expression(_exact_ufl(msh, ScalarType(t0)), V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    u_init = fem.Function(V)
    u_init.x.array[:] = u_n.x.array

    u_bc = fem.Function(V)
    f_fun = fem.Function(V)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    dt_c = fem.Constant(msh, ScalarType(dt))
    a = (u * v + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    u_h = fem.Function(V)
    total_iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt

        u_bc.interpolate(fem.Expression(_exact_ufl(msh, ScalarType(t)), V.element.interpolation_points))

        u_exact_t = _exact_ufl(msh, ScalarType(t))
        f_ufl = ufl.diff(u_exact_t, t) - ufl.div(_kappa_ufl(msh) * ufl.grad(u_exact_t))
        f_fun.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

        with b.localForm() as b_loc:
            b_loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iterations += solver.getIterationNumber()

        u_n.x.array[:] = u_h.x.array

    u_exact_final = fem.Function(V)
    u_exact_final.interpolate(fem.Expression(_exact_ufl(msh, ScalarType(t_end)), V.element.interpolation_points))
    l2_error = _compute_l2_error(u_h, u_exact_final)

    u_grid = _sample_function(u_h, case_spec["output"]["grid"])
    u_initial_grid = _sample_function(u_init, case_spec["output"]["grid"])

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(1e-10),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}
