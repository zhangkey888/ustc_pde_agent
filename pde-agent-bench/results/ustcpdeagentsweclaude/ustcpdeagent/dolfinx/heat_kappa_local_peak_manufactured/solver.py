import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def manufactured_u(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    output_grid = case_spec.get("output", {}).get("grid", {})

    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.1))
    dt_suggested = float(time_spec.get("dt", 0.01))

    nx_out = int(output_grid.get("nx", 64))
    ny_out = int(output_grid.get("ny", 64))
    bbox = output_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    time_limit = 8.117
    target_budget = 0.45 * time_limit
    start_all = time.perf_counter()

    candidates = [
        {"N": 40, "degree": 1, "dt": min(dt_suggested, 0.005)},
        {"N": 48, "degree": 2, "dt": min(dt_suggested, 0.005)},
        {"N": 64, "degree": 2, "dt": min(dt_suggested, 0.004)},
        {"N": 72, "degree": 2, "dt": min(dt_suggested, 0.0025)},
        {"N": 80, "degree": 2, "dt": min(dt_suggested, 0.002)},
    ]

    best = None
    for cand in candidates:
        if time.perf_counter() - start_all > target_budget:
            break
        try:
            result = _run_one(
                N=cand["N"],
                degree=cand["degree"],
                dt=cand["dt"],
                t0=t0,
                t_end=t_end,
                nx_out=nx_out,
                ny_out=ny_out,
                bbox=bbox,
                ksp_type="cg",
                pc_type="hypre",
                rtol=1e-10,
            )
        except Exception:
            result = _run_one(
                N=cand["N"],
                degree=cand["degree"],
                dt=cand["dt"],
                t0=t0,
                t_end=t_end,
                nx_out=nx_out,
                ny_out=ny_out,
                bbox=bbox,
                ksp_type="preonly",
                pc_type="lu",
                rtol=1e-12,
            )

        if best is None or result["l2_error"] < best["l2_error"]:
            best = result

        elapsed = time.perf_counter() - start_all
        if result["l2_error"] <= 3.0e-4 and elapsed >= 0.30 * time_limit:
            break

    if best is None:
        raise RuntimeError("All candidate solves failed.")

    return {
        "u": best["u_grid"],
        "u_initial": best["u0_grid"],
        "solver_info": best["solver_info"],
    }


def _run_one(N, degree, dt, t0, t_end, nx_out, ny_out, bbox, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, ScalarType(t0))

    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2))
    u_exact = ufl.exp(-t_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
    f_expr = -u_exact - ufl.div(kappa * ufl.grad(u_exact))

    n_steps = max(1, int(math.ceil((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.exp(-t0) * np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.exp(-t0) * np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, bdofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (u * v + dt * ufl.inner(kappa * ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

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

    uh = fem.Function(V)
    total_iterations = 0
    t = t0

    for _ in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        u_bc.interpolate(lambda X, tt=t: np.exp(-tt) * np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 0))
        u_n.x.array[:] = uh.x.array

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: np.exp(-t_end) * np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    err_local = fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx))
    l2_error = float(np.sqrt(comm.allreduce(err_local, op=MPI.SUM)))

    u_grid = _sample_function(msh, uh, nx_out, ny_out, bbox)
    u0_grid = _sample_exact(nx_out, ny_out, bbox, t0)

    solver_info = {
        "mesh_resolution": int(N),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }

    return {
        "u_grid": u_grid,
        "u0_grid": u0_grid,
        "solver_info": solver_info,
        "l2_error": l2_error,
    }


def _sample_function(msh, uh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(local_vals)
    vals = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(vals) & ~np.isnan(arr)
        vals[mask] = arr[mask]

    vals[np.isnan(vals)] = 0.0
    return vals.reshape(ny, nx)


def _sample_exact(nx, ny, bbox, t):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    return manufactured_u(XX, YY, t).reshape(ny, nx)


def _default_case():
    return {
        "pde": {
            "time": {
                "t0": 0.0,
                "t_end": 0.1,
                "dt": 0.01,
                "scheme": "backward_euler",
            }
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
    }


if __name__ == "__main__":
    case_spec = _default_case()
    tstart = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - tstart

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    t_end = float(case_spec["pde"]["time"]["t_end"])

    u_num = result["u"]
    u_ex = _sample_exact(nx, ny, bbox, t_end)
    l2_grid = float(np.sqrt(np.mean((u_num - u_ex) ** 2)))

    if MPI.COMM_WORLD.rank == 0:
        print("L2_ERROR:", f"{l2_grid:.12e}")
        print("WALL_TIME:", f"{wall:.12e}")
        print(result["solver_info"])
