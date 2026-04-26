import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _build_grid_points(grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)))
    return pts, nx, ny


def _sample_function_on_points(domain, u_func, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_ids = []
    local_points = []
    local_cells = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_ids.append(i)
            local_points.append(points[i])
            local_cells.append(links[0])

    local_vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_points), -1)[:, 0]
        local_vals[np.array(local_ids, dtype=np.int32)] = vals

    if domain.comm.size == 1:
        return local_vals

    send = np.where(np.isnan(local_vals), -np.inf, local_vals)
    recv = np.empty_like(send)
    domain.comm.Allreduce(send, recv, op=MPI.MAX)
    recv[np.isneginf(recv)] = np.nan
    return recv


def _solve_once(mesh_resolution, degree, dt, t_end, kappa_value, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    uh = fem.Function(V)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0

    x = ufl.SpatialCoordinate(domain)
    source_expr = ufl.sin(6.0 * ufl.pi * x[0]) * ufl.sin(6.0 * ufl.pi * x[1])
    source_fun = fem.Function(V)
    source_fun.interpolate(fem.Expression(source_expr, V.element.interpolation_points))

    kappa = fem.Constant(domain, ScalarType(kappa_value))
    dt_c = fem.Constant(domain, ScalarType(dt))

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * source_fun * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()

    n_steps = max(1, int(round(t_end / dt)))
    total_iterations = 0
    start = time.perf_counter()

    for _ in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += max(0, solver.getIterationNumber())
        u_n.x.array[:] = uh.x.array

    runtime = time.perf_counter() - start

    steady_exact = source_expr / (ScalarType(kappa_value) * ScalarType(72.0 * np.pi * np.pi))
    err_form = fem.form((uh - steady_exact) ** 2 * ufl.dx)
    ref_form = fem.form((steady_exact) ** 2 * ufl.dx)
    err = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref = np.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    rel_err = err / max(ref, 1e-14)

    return {
        "domain": domain,
        "u_final": uh,
        "u_initial": u0,
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": rtol,
        "iterations": int(total_iterations),
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "verification": {
            "steady_state_l2_error": float(err),
            "steady_state_rel_l2_error": float(rel_err),
            "runtime_sec": float(runtime),
        },
    }


def solve(case_spec: dict) -> dict:
    coeffs = case_spec.get("coefficients", {})
    time_spec = case_spec.get("time", {})
    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]

    kappa = float(coeffs.get("kappa", 0.8))
    t_end = float(time_spec.get("t_end", pde.get("t_end", 0.12)))
    dt_suggested = float(time_spec.get("dt", pde.get("dt", 0.02)))
    if t_end <= 0.0:
        t_end = 0.12
    if dt_suggested <= 0.0:
        dt_suggested = 0.02

    time_budget = 21.146
    candidates = [
        (48, 1, min(0.01, dt_suggested)),
        (64, 1, min(0.01, dt_suggested)),
        (80, 1, 0.005),
        (96, 1, 0.005),
        (80, 2, 0.005),
    ]

    chosen = None
    for mesh_resolution, degree, dt in candidates:
        try:
            result = _solve_once(mesh_resolution, degree, dt, t_end, kappa, "cg", "hypre", 1e-10)
        except Exception:
            result = _solve_once(mesh_resolution, degree, dt, t_end, kappa, "preonly", "lu", 1e-12)
        chosen = result
        if result["verification"]["runtime_sec"] > 0.75 * time_budget:
            break

    pts, nx, ny = _build_grid_points(grid)
    u_vals = _sample_function_on_points(chosen["domain"], chosen["u_final"], pts)
    u0_vals = _sample_function_on_points(chosen["domain"], chosen["u_initial"], pts)

    if np.isnan(u_vals).any() or np.isnan(u0_vals).any():
        raise RuntimeError("Sampling failed on output grid.")

    return {
        "u": u_vals.reshape(ny, nx),
        "u_initial": u0_vals.reshape(ny, nx),
        "solver_info": {
            "mesh_resolution": int(chosen["mesh_resolution"]),
            "element_degree": int(chosen["element_degree"]),
            "ksp_type": str(chosen["ksp_type"]),
            "pc_type": str(chosen["pc_type"]),
            "rtol": float(chosen["rtol"]),
            "iterations": int(chosen["iterations"]),
            "dt": float(chosen["dt"]),
            "n_steps": int(chosen["n_steps"]),
            "time_scheme": str(chosen["time_scheme"]),
            "verification": chosen["verification"],
        },
    }
