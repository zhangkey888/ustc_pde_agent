import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _exact_expr(x, t):
    return np.exp(-0.5 * t) * np.sin(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])


def _make_case_defaults(case_spec):
    out = dict(case_spec) if case_spec is not None else {}
    out.setdefault("output", {})
    out["output"].setdefault("grid", {})
    out["output"]["grid"].setdefault("nx", 64)
    out["output"]["grid"].setdefault("ny", 64)
    out["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    out.setdefault("pde", {})
    out["pde"].setdefault("time", {})
    out["pde"]["time"].setdefault("t0", 0.0)
    out["pde"]["time"].setdefault("t_end", 0.2)
    out["pde"]["time"].setdefault("dt", 0.02)
    return out


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(global_vals) & ~np.isnan(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            raise RuntimeError("Failed to evaluate solution on all requested grid points.")
        return global_vals.reshape(ny, nx)
    return None


def solve(case_spec: dict) -> dict:
    case_spec = _make_case_defaults(case_spec)
    comm = MPI.COMM_WORLD
    rank = comm.rank

    kappa = 0.1
    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.2))
    dt_suggested = float(case_spec["pde"]["time"].get("dt", 0.02))

    # Accuracy-focused choice well within small runtime budget
    mesh_resolution = 96
    element_degree = 2
    dt = min(dt_suggested, 0.005)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt_suggested

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(t0))
    dt_c = fem.Constant(domain, ScalarType(dt))
    kappa_c = fem.Constant(domain, ScalarType(kappa))

    u_exact_ufl = ufl.exp(-0.5 * t_c) * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f_ufl = (-0.5 + 5.0 * (ufl.pi ** 2) * kappa_c) * u_exact_ufl

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _exact_expr(X, t0))
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _exact_expr(X, t0))

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    solver.setFromOptions()

    uh = fem.Function(V)
    iterations = 0

    for step in range(1, n_steps + 1):
        t_c.value = ScalarType(t0 + step * dt)
        u_bc.interpolate(lambda X, tt=float(t0 + step * dt): _exact_expr(X, tt))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_initial_grid = _sample_on_grid(domain, fem.Function(V), case_spec["output"]["grid"])
    if rank == 0:
        xmin, xmax, ymin, ymax = case_spec["output"]["grid"]["bbox"]
        nx = int(case_spec["output"]["grid"]["nx"])
        ny = int(case_spec["output"]["grid"]["ny"])
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        u_initial_grid = np.exp(-0.5 * t0) * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)

    u_grid = _sample_on_grid(domain, uh, case_spec["output"]["grid"])

    err_form = fem.form((uh - u_exact_ufl) ** 2 * ufl.dx)
    l2_error_local = fem.assemble_scalar(err_form)
    l2_error = math.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1e-10,
        "iterations": int(iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "l2_error": float(l2_error),
    }

    if rank == 0:
        return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.2, "dt": 0.02}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0_wall = time.time()
    result = solve(case_spec)
    wall = time.time() - t0_wall
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {result['solver_info']['l2_error']:.12e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(f"U_SHAPE: {result['u'].shape}")
