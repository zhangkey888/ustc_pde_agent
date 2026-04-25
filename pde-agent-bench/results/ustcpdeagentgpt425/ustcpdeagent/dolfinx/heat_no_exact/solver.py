import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _time_params(case_spec: dict):
    pde = case_spec.get("pde", {})
    ts = pde.get("time", {})
    t0 = float(ts.get("t0", 0.0))
    t_end = float(ts.get("t_end", 0.1))
    dt = float(ts.get("dt", 0.02))
    scheme = ts.get("scheme", "backward_euler")
    return t0, t_end, dt, scheme


def _grid_spec(case_spec: dict):
    out = case_spec.get("output", {}).get("grid", {})
    return {
        "nx": int(out.get("nx", 64)),
        "ny": int(out.get("ny", 64)),
        "bbox": out.get("bbox", [0.0, 1.0, 0.0, 1.0]),
    }


def _build_problem(n: int, degree: int, dt: float, t0: float, t_end: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(msh, ScalarType(1.0))
    dt_c = fem.Constant(msh, ScalarType(dt))

    f = fem.Function(V)
    f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    n_steps = max(1, int(round((t_end - t0) / dt)))
    actual_dt = (t_end - t0) / n_steps
    dt_c.value = ScalarType(actual_dt)

    a = (u * v + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setTolerances(rtol=1e-10)

    uh = fem.Function(V)

    return {
        "mesh": msh,
        "V": V,
        "u_n": u_n,
        "uh": uh,
        "bc": bc,
        "a_form": a_form,
        "L_form": L_form,
        "A": A,
        "b": b,
        "solver": solver,
        "dt": float(actual_dt),
        "n_steps": int(n_steps),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
    }


def _advance(state):
    total_iterations = 0
    b = state["b"]
    for _ in range(state["n_steps"]):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, state["L_form"])
        petsc.apply_lifting(b, [state["a_form"]], bcs=[[state["bc"]]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [state["bc"]])
        state["solver"].solve(b, state["uh"].x.petsc_vec)
        state["uh"].x.scatter_forward()
        total_iterations += int(state["solver"].getIterationNumber())
        state["u_n"].x.array[:] = state["uh"].x.array[:]
    return total_iterations


def _sample_on_grid(msh, u_fun, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    idx = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            idx.append(i)

    if points_on_proc:
        out = u_fun.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells, dtype=np.int32))
        vals[np.array(idx, dtype=np.int32)] = np.asarray(out).reshape(-1)

    gathered = msh.comm.allgather(vals)
    final = np.full_like(vals, np.nan)
    for g in gathered:
        mask = np.isnan(final) & ~np.isnan(g)
        final[mask] = g[mask]
    final = np.nan_to_num(final, nan=0.0)
    return final.reshape((ny, nx))


def solve(case_spec: dict) -> dict:
    t0, t_end, dt_suggested, scheme = _time_params(case_spec)
    if scheme != "backward_euler":
        scheme = "backward_euler"
    grid = _grid_spec(case_spec)

    start = time.perf_counter()
    candidates = [
        (24, min(dt_suggested, 0.01)),
        (36, min(dt_suggested, 0.01)),
        (48, min(dt_suggested / 2.0, 0.005)),
    ]

    chosen_state = None
    chosen_iterations = 0
    u_initial_grid = None

    for i, (n, dt) in enumerate(candidates):
        state = _build_problem(n=n, degree=1, dt=dt, t0=t0, t_end=t_end)
        u0_grid = _sample_on_grid(state["mesh"], state["u_n"], grid)
        its = _advance(state)
        chosen_state = state
        chosen_iterations = its
        u_initial_grid = u0_grid
        elapsed = time.perf_counter() - start
        if i < len(candidates) - 1 and elapsed > 8.0:
            break

    u_grid = _sample_on_grid(chosen_state["mesh"], chosen_state["uh"], grid)

    verify_n = max(12, chosen_state["mesh_resolution"] // 2)
    verify_dt = min(0.02, 2.0 * chosen_state["dt"])
    verify_state = _build_problem(n=verify_n, degree=1, dt=verify_dt, t0=t0, t_end=t_end)
    _advance(verify_state)
    u_grid_coarse = _sample_on_grid(verify_state["mesh"], verify_state["uh"], grid)
    rmse = float(np.sqrt(np.mean((u_grid - u_grid_coarse) ** 2)))

    solver_info = {
        "mesh_resolution": int(chosen_state["mesh_resolution"]),
        "element_degree": int(chosen_state["element_degree"]),
        "ksp_type": str(chosen_state["ksp_type"]),
        "pc_type": str(chosen_state["pc_type"]),
        "rtol": float(chosen_state["rtol"]),
        "iterations": int(chosen_iterations),
        "dt": float(chosen_state["dt"]),
        "n_steps": int(chosen_state["n_steps"]),
        "time_scheme": scheme,
        "verification_rmse_vs_coarse": rmse,
    }

    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"}},
        "output": {"grid": {"nx": 16, "ny": 16, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
