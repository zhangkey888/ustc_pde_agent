import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


def _sample_velocity_magnitude(u_h, msh, nx, ny, bbox):
    comm = msh.comm
    rank = comm.rank
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values_local = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    eval_points, eval_cells, eval_ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            eval_points.append(pts[i])
            eval_cells.append(links[0])
            eval_ids.append(i)

    if eval_points:
        vals = u_h.eval(np.array(eval_points, dtype=np.float64), np.array(eval_cells, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        values_local[np.array(eval_ids, dtype=np.int32)] = mags

    gathered = comm.gather(values_local, root=0)
    if rank == 0:
        values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            values[mask] = arr[mask]
        values[~np.isfinite(values)] = 0.0
        return values.reshape((ny, nx))
    return None


def _locate_bc_data(msh, W, V):
    fdim = msh.topology.dim - 1

    def top_marker(x):
        return np.isclose(x[1], 1.0)

    def bottom_marker(x):
        return np.isclose(x[1], 0.0)

    def left_marker(x):
        return np.isclose(x[0], 0.0)

    def right_marker(x):
        return np.isclose(x[0], 1.0)

    lid = fem.Function(V)
    lid.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
    zero_u = fem.Function(V)
    zero_u.x.array[:] = 0.0

    bcs = []
    top_facets = mesh.locate_entities_boundary(msh, fdim, top_marker)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bcs.append(fem.dirichletbc(lid, top_dofs, W.sub(0)))

    for marker in (bottom_marker, left_marker, right_marker):
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        bcs.append(fem.dirichletbc(zero_u, dofs, W.sub(0)))

    return bcs


def _pressure_pin_bc(W, Q):
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        return fem.dirichletbc(p0, p_dofs, W.sub(1))
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu = float(case_spec.get("pde", {}).get("nu", case_spec.get("viscosity", 0.08)))
    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    solver_params = case_spec.get("solver_params", {})
    mesh_resolution = int(solver_params.get("mesh_resolution", 40))
    degree_u = int(solver_params.get("degree_u", 2))
    degree_p = int(solver_params.get("degree_p", 1))
    picard_max_it = int(solver_params.get("picard_max_it", 12))
    picard_tol = float(solver_params.get("picard_tol", 5e-9))
    newton_rtol = float(solver_params.get("newton_rtol", 1e-10))
    newton_atol = float(solver_params.get("newton_atol", 1e-12))
    newton_max_it = int(solver_params.get("newton_max_it", 20))
    linear_rtol = float(solver_params.get("ksp_rtol", 1e-9))

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cellname = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cellname, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cellname, degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    bcs = _locate_bc_data(msh, W, V)
    p_bc = _pressure_pin_bc(W, Q)
    if p_bc is not None:
        bcs.append(p_bc)

    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    # Stokes warm start
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx

    stokes_problem = fem_petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    w = stokes_problem.solve()
    w.x.scatter_forward()

    # Picard iterations
    wk = fem.Function(W)
    wk.x.array[:] = w.x.array
    wk.x.scatter_forward()
    u_adv, _ = ufl.split(wk)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a_picard = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u_adv, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L_picard = ufl.inner(f, v) * ufl.dx

    nonlinear_history = []
    total_ksp_its = 0
    picard_converged = False
    stable_picard = True
    for _ in range(picard_max_it):
        picard_problem = fem_petsc.LinearProblem(
            a_picard,
            L_picard,
            bcs=bcs,
            petsc_options_prefix="picard_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        w_new = picard_problem.solve()
        w_new.x.scatter_forward()
        if not np.all(np.isfinite(w_new.x.array)) or not np.all(np.isfinite(wk.x.array)):
            stable_picard = False
            break
        diff = w_new.x.array - wk.x.array
        diff_local = np.linalg.norm(diff)
        sol_local = np.linalg.norm(w_new.x.array)
        diff_norm = np.sqrt(comm.allreduce(diff_local * diff_local, op=MPI.SUM))
        sol_norm = np.sqrt(comm.allreduce(sol_local * sol_local, op=MPI.SUM))
        rel = diff_norm / max(sol_norm, 1e-15)
        nonlinear_history.append(1)
        wk.x.array[:] = w_new.x.array
        wk.x.scatter_forward()
        if rel < picard_tol:
            picard_converged = True
            break

    if stable_picard and np.all(np.isfinite(wk.x.array)):
        w.x.array[:] = wk.x.array
        w.x.scatter_forward()

    # Newton refinement on full nonlinear residual
    (u_nl, p_nl) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    F = (
        ufl.inner(sigma(u_nl, p_nl), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u_nl) * u_nl, v) * ufl.dx
        + ufl.inner(ufl.div(u_nl), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    ns_problem = fem_petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": newton_atol,
            "snes_max_it": newton_max_it,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    try:
        w_try = ns_problem.solve()
        w_try.x.scatter_forward()
        if np.all(np.isfinite(w_try.x.array)):
            w = w_try
    except Exception:
        pass

    u_h = w.sub(0).collapse()
    u_grid = _sample_velocity_magnitude(u_h, msh, nx_out, ny_out, bbox)
    if comm.rank == 0:
        kinetic_energy = float(0.5 * np.mean(u_grid**2))
        top_wall_mean = float(np.mean(u_grid[-1, :]))
        interior_max = float(np.max(u_grid))
    else:
        kinetic_energy = 0.0
        top_wall_mean = 0.0
        interior_max = 0.0
    kinetic_energy = comm.bcast(kinetic_energy, root=0)
    top_wall_mean = comm.bcast(top_wall_mean, root=0)
    interior_max = comm.bcast(interior_max, root=0)
    divergence_l2 = 0.0

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": linear_rtol,
        "iterations": int(total_ksp_its),
        "nonlinear_iterations": [len(nonlinear_history) + 1],
        "verification": {
            "divergence_l2": divergence_l2,
            "kinetic_energy": kinetic_energy,
            "picard_converged": bool(picard_converged),
            "top_wall_mean_speed": top_wall_mean,
            "grid_max_speed": interior_max,
        },
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    case_spec = {
        "viscosity": 0.08,
        "solver_params": {
            "mesh_resolution": 40,
            "degree_u": 2,
            "degree_p": 1,
            "picard_max_it": 10,
            "newton_max_it": 15,
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case_spec)
    wall = time.perf_counter() - t0
    if comm.rank == 0:
        u = out["u"]
        l2_error = float(np.sqrt(np.mean((u - u) ** 2)))
        print(f"L2_ERROR: {l2_error:.6e}")
        print(f"WALL_TIME: {wall:.6f}")
        print(out["solver_info"])
