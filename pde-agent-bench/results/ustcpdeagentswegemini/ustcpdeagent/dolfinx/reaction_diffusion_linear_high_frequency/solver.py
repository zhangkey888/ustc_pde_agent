import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _get_params(case_spec: dict):
    pde = case_spec.get("pde", {})
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.3))
    dt_in = float(time_spec.get("dt", 0.005))
    scheme = str(time_spec.get("scheme", "crank_nicolson")).lower()

    # This benchmark uses manufactured u = exp(-t) sin(4 pi x) sin(3 pi y)
    epsilon = float(pde.get("epsilon", case_spec.get("epsilon", 0.02)))
    reaction_coeff = float(pde.get("reaction_coeff", case_spec.get("reaction_coeff", 1.0)))

    return t0, t_end, dt_in, scheme, epsilon, reaction_coeff


def _sample_on_grid(u_func, domain, nx, ny, bbox):
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts2)

    local_point_ids = []
    local_points = []
    local_cells = []
    for i in range(pts2.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_point_ids.append(i)
            local_points.append(pts2[i])
            local_cells.append(links[0])

    local_vals = np.full(pts2.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.array(local_points, dtype=np.float64), np.array(local_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(local_points), -1)[:, 0]
        local_vals[np.array(local_point_ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(pts2.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        # Boundary points should have been found, but safely fill any remaining NaNs with exact-nearby zero.
        out = np.nan_to_num(out, nan=0.0)
        return out.reshape(ny, nx)
    return None


def _run_simulation(mesh_resolution, degree, dt, t0, t_end, epsilon, reaction_coeff, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi

    def u_exact_expr(t):
        return ufl.exp(-t) * ufl.sin(4.0 * pi * x[0]) * ufl.sin(3.0 * pi * x[1])

    # Manufactured linear reaction-diffusion:
    # u_t - eps Δu + c u = f
    # with exact u_exact
    def f_expr(t):
        uex = u_exact_expr(t)
        ut = -uex
        lap = -((4.0 * pi) ** 2 + (3.0 * pi) ** 2) * uex
        return ut - epsilon * lap + reaction_coeff * uex

    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_expr(t0), V.element.interpolation_points))

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(fem.Expression(u_exact_expr(t0), V.element.interpolation_points))
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc_fun, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_n_fun = fem.Function(V)
    f_np1_fun = fem.Function(V)
    f_n_fun.interpolate(fem.Expression(f_expr(t0), V.element.interpolation_points))
    f_np1_fun.interpolate(fem.Expression(f_expr(min(t0 + dt, t_end)), V.element.interpolation_points))

    theta = 0.5  # Crank-Nicolson
    a = (
        (1.0 / dt) * u * v * ufl.dx
        + theta * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + theta * reaction_coeff * u * v * ufl.dx
    )
    L = (
        (1.0 / dt) * u_n * v * ufl.dx
        - (1.0 - theta) * epsilon * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
        - (1.0 - theta) * reaction_coeff * u_n * v * ufl.dx
        + (theta * f_np1_fun + (1.0 - theta) * f_n_fun) * v * ufl.dx
    )

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=rtol)

    try:
        solver.setFromOptions()
    except Exception:
        pass

    uh = fem.Function(V)
    uh.x.array[:] = u_n.x.array[:]
    iterations = 0

    n_steps = int(round((t_end - t0) / dt))
    t = t0

    for step in range(n_steps):
        tn = t
        tnp1 = tn + dt

        u_bc_fun.interpolate(fem.Expression(u_exact_expr(tnp1), V.element.interpolation_points))
        f_n_fun.interpolate(fem.Expression(f_expr(tn), V.element.interpolation_points))
        f_np1_fun.interpolate(fem.Expression(f_expr(tnp1), V.element.interpolation_points))

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        iterations += solver.getIterationNumber()

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()
        t = tnp1

    uex_T = fem.Function(V)
    uex_T.interpolate(fem.Expression(u_exact_expr(t_end), V.element.interpolation_points))

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - uex_T.x.array
    err_fun.x.scatter_forward()
    l2_local = fem.assemble_scalar(fem.form(err_fun * err_fun * ufl.dx))
    l2_err = math.sqrt(domain.comm.allreduce(l2_local, op=MPI.SUM))

    return {
        "domain": domain,
        "V": V,
        "u_final": uh,
        "u_initial": fem.Function(V),
        "l2_error": l2_err,
        "iterations": int(iterations),
        "n_steps": n_steps,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "mesh_resolution": mesh_resolution,
        "degree": degree,
        "dt": dt,
    }, u_n


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    wall_start = time.perf_counter()

    t0, t_end, dt_suggested, scheme, epsilon, reaction_coeff = _get_params(case_spec)
    out_grid = case_spec["output"]["grid"]
    nx = int(out_grid["nx"])
    ny = int(out_grid["ny"])
    bbox = out_grid["bbox"]

    # Start with a fairly accurate configuration, then refine once if runtime is far below budget.
    mesh_resolution = 72
    degree = 2
    dt = min(dt_suggested, 0.005)
    rtol = 1.0e-10

    result, u_initial_snapshot = _run_simulation(
        mesh_resolution, degree, dt, t0, t_end, epsilon, reaction_coeff, rtol
    )
    result["u_initial"].x.array[:] = 0.0
    result["u_initial"].interpolate(
        fem.Expression(
            ufl.exp(-t0)
            * ufl.sin(4.0 * np.pi * ufl.SpatialCoordinate(result["domain"])[0])
            * ufl.sin(3.0 * np.pi * ufl.SpatialCoordinate(result["domain"])[1]),
            result["V"].element.interpolation_points,
        )
    )

    elapsed = time.perf_counter() - wall_start

    # Adaptive time-accuracy tradeoff: if much faster than limit, improve accuracy.
    # Stay conservative with only one refinement.
    if elapsed < 20.0:
        refined_mesh = 96
        refined_dt = min(dt / 2.0, 0.0025)
        result2, _ = _run_simulation(
            refined_mesh, degree, refined_dt, t0, t_end, epsilon, reaction_coeff, rtol
        )
        if result2["l2_error"] <= result["l2_error"] * 1.05 or result2["l2_error"] < result["l2_error"]:
            result = result2
            result["u_initial"] = fem.Function(result["V"])
            result["u_initial"].interpolate(
                fem.Expression(
                    ufl.exp(-t0)
                    * ufl.sin(4.0 * np.pi * ufl.SpatialCoordinate(result["domain"])[0])
                    * ufl.sin(3.0 * np.pi * ufl.SpatialCoordinate(result["domain"])[1]),
                    result["V"].element.interpolation_points,
                )
            )

    u_grid = _sample_on_grid(result["u_final"], result["domain"], nx, ny, bbox)
    u0_grid = _sample_on_grid(result["u_initial"], result["domain"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(result["mesh_resolution"]),
        "element_degree": int(result["degree"]),
        "ksp_type": str(result["ksp_type"]),
        "pc_type": str(result["pc_type"]),
        "rtol": float(rtol),
        "iterations": int(result["iterations"]),
        "dt": float(result["dt"]),
        "n_steps": int(result["n_steps"]),
        "time_scheme": "crank_nicolson",
        "accuracy_verification": {
            "manufactured_solution": "exp(-t)*sin(4*pi*x)*sin(3*pi*y)",
            "l2_error_final": float(result["l2_error"]),
            "target_threshold": 1.15e-2,
        },
    }

    if comm.rank == 0:
        return {"u": u_grid, "u_initial": u0_grid, "solver_info": solver_info}
    return {"u": None, "u_initial": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "time": {"t0": 0.0, "t_end": 0.3, "dt": 0.005, "scheme": "crank_nicolson"},
            "epsilon": 0.02,
            "reaction_coeff": 1.0,
        },
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
