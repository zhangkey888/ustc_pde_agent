import math
import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _exact_values(x, t):
    return np.exp(-t) * np.exp(5.0 * x[1]) * np.sin(np.pi * x[0])


def _make_bc(V, t):
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x, tt=t: _exact_values(x, tt))
    return fem.dirichletbc(u_bc, dofs), u_bc


def _sample_function(u_fun, grid_spec):
    msh = u_fun.function_space.mesh
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_fun.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        local_vals[np.array(eval_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution at all requested grid points.")
        merged = merged.reshape((ny, nx))
    else:
        merged = None

    merged = comm.bcast(merged, root=0)
    return merged


def _run_case(nx, degree, dt, t_end, kappa):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    u_n = fem.Function(V)
    u_n.interpolate(lambda x: _exact_values(x, 0.0))
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    n_steps = max(1, int(round(t_end / dt)))
    dt = t_end / n_steps

    bc, u_bc = _make_bc(V, dt)
    f_expr = (
        -ufl.exp(-dt) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
        - kappa * ufl.exp(-dt) * ufl.exp(5.0 * x[1]) * (25.0 - np.pi**2) * ufl.sin(ufl.pi * x[0])
    )

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=10000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        t = step * dt
        u_bc.interpolate(lambda X, tt=t: _exact_values(X, tt))
        u_bc.x.scatter_forward()

        f_expr = (
            -ufl.exp(-t) * ufl.exp(5.0 * x[1]) * ufl.sin(ufl.pi * x[0])
            - kappa * ufl.exp(-t) * ufl.exp(5.0 * x[1]) * (25.0 - np.pi**2) * ufl.sin(ufl.pi * x[0])
        )
        L_form = fem.form((u_n * v + dt * f_expr * v) * ufl.dx)
        b = petsc.create_vector(L_form.function_spaces)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        try:
            solver.solve(b, uh.x.petsc_vec)
        except Exception:
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.setOperators(A)
            solver.solve(b, uh.x.petsc_vec)

        uh.x.scatter_forward()
        total_iterations += int(solver.getIterationNumber())

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    elapsed = time.perf_counter() - start

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda X: _exact_values(X, t_end))
    err_form = fem.form((uh - u_ex) ** 2 * ufl.dx)
    l2_error = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))

    return {
        "mesh_resolution": int(nx),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
        "verification_l2_error": float(l2_error),
        "wall_time": float(elapsed),
        "solution": uh,
        "space": V,
    }


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = case_spec.get("coefficients", {})
    out_grid = case_spec["output"]["grid"]

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.08))
    dt0 = float(pde.get("dt", 0.008))
    if t_end <= t0:
        t0, t_end = 0.0, 0.08
    t_final = t_end - t0

    kappa = float(coeffs.get("kappa", 1.0))
    time_budget = 18.264

    candidates = [
        (40, 2, dt0),
        (56, 2, dt0 / 2.0),
        (72, 2, dt0 / 2.0),
        (88, 2, dt0 / 4.0),
    ]

    best = None
    for nx, degree, dt in candidates:
        result = _run_case(nx, degree, dt, t_final, kappa)
        if result["wall_time"] > time_budget:
            continue
        if best is None:
            best = result
        else:
            if result["verification_l2_error"] < best["verification_l2_error"]:
                best = result

    if best is None:
        best = _run_case(40, 2, dt0, t_final, kappa)

    uh = best["solution"]
    u_grid = _sample_function(uh, out_grid)

    V = best["space"]
    u0_fun = fem.Function(V)
    u0_fun.interpolate(lambda x: _exact_values(x, 0.0))
    u0_fun.x.scatter_forward()
    u0_grid = _sample_function(u0_fun, out_grid)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "dt": best["dt"],
        "n_steps": best["n_steps"],
        "time_scheme": best["time_scheme"],
    }

    return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {
        "pde": {"t0": 0.0, "t_end": 0.08, "dt": 0.008, "time": True},
        "coefficients": {"kappa": 1.0},
        "output": {"grid": {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
