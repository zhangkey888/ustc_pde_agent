import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _diagnosis_and_method_cards():
    return {
        "DIAGNOSIS": {
            "equation_type": "navier_stokes",
            "spatial_dim": 2,
            "domain_geometry": "rectangle",
            "unknowns": "vector+scalar",
            "coupling": "saddle_point",
            "linearity": "nonlinear",
            "time_dependence": "steady",
            "stiffness": "N/A",
            "dominant_physics": "mixed",
            "peclet_or_reynolds": "moderate",
            "solution_regularity": "smooth",
            "bc_type": "all_dirichlet",
            "special_notes": "manufactured_solution",
        },
        "METHOD": {
            "spatial_method": "fem",
            "element_or_basis": "Taylor-Hood_P2P1",
            "stabilization": "none",
            "time_method": "none",
            "nonlinear_solver": "newton",
            "linear_solver": "gmres",
            "preconditioner": "lu",
            "special_treatment": "pressure_pinning",
            "pde_skill": "navier_stokes",
        },
    }


def _build_exact_ufl(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector(
        [
            pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
            + pi * ufl.cos(4 * pi * x[1]) * ufl.sin(2 * pi * x[0]),
            -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
            - (pi / 2) * ufl.cos(2 * pi * x[0]) * ufl.sin(4 * pi * x[1]),
        ]
    )
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(2 * pi * x[1])
    return u_exact, p_exact


def _forcing_ufl(msh, nu):
    u_exact, p_exact = _build_exact_ufl(msh)
    return ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)


def _exact_velocity_callable():
    pi = np.pi

    def g(X):
        x = X[0]
        y = X[1]
        return np.vstack(
            [
                pi * np.cos(pi * y) * np.sin(pi * x)
                + pi * np.cos(4 * pi * y) * np.sin(2 * pi * x),
                -pi * np.cos(pi * x) * np.sin(pi * y)
                - 0.5 * pi * np.cos(2 * pi * x) * np.sin(4 * pi * y),
            ]
        )

    return g


def _sample_function_on_grid(func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_ids_arr = np.array(local_ids, dtype=np.int32)
    local_vals = None
    if len(local_points) > 0:
        local_vals = func.eval(
            np.array(local_points, dtype=np.float64),
            np.array(local_cells, dtype=np.int32),
        )

    gathered_ids = msh.comm.gather(local_ids_arr, root=0)
    gathered_vals = msh.comm.gather(local_vals, root=0)

    out = None
    if msh.comm.rank == 0:
        value_shape = tuple(func.function_space.element.value_shape)
        if len(value_shape) == 0:
            out = np.empty((pts.shape[0],), dtype=np.float64)
        else:
            out = np.empty((pts.shape[0], int(np.prod(value_shape))), dtype=np.float64)
        filled = np.zeros(pts.shape[0], dtype=bool)
        for ids, vals in zip(gathered_ids, gathered_vals):
            if ids is None or len(ids) == 0:
                continue
            out[ids] = vals
            filled[ids] = True
        if not np.all(filled):
            missing = np.where(~filled)[0]
            raise RuntimeError(f"Failed to evaluate all grid points; missing {len(missing)} points")
    return msh.comm.bcast(out, root=0)


def _compute_errors(w, msh):
    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    u_exact, p_exact = _build_exact_ufl(msh)
    Vh = uh.function_space
    Qh = ph.function_space

    u_ex = fem.Function(Vh)
    u_ex.interpolate(fem.Expression(u_exact, Vh.element.interpolation_points))
    p_ex = fem.Function(Qh)
    p_ex.interpolate(fem.Expression(p_exact, Qh.element.interpolation_points))

    e_u = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    e_p = fem.form((ph - p_ex) * (ph - p_ex) * ufl.dx)

    u_l2 = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(e_u), op=MPI.SUM))
    p_l2 = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(e_p), op=MPI.SUM))
    return u_l2, p_l2


def _solve_on_mesh(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    gdim = msh.geometry.dim
    nu_value = 0.1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = _forcing_ufl(msh, nu)

    u_bc_fun = fem.Function(V)
    u_bc_fun.interpolate(_exact_velocity_callable())

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    w.x.array[:] = 0.0
    w.x.scatter_forward()

    us, ps = ufl.TrialFunctions(W)
    vs, qs = ufl.TestFunctions(W)

    a_stokes = (
        nu * ufl.inner(ufl.grad(us), ufl.grad(vs)) * ufl.dx
        - ufl.inner(ps, ufl.div(vs)) * ufl.dx
        + ufl.inner(ufl.div(us), qs) * ufl.dx
    )
    L_stokes = ufl.inner(f, vs) * ufl.dx

    stokes_problem = petsc.LinearProblem(
        a_stokes,
        L_stokes,
        bcs=bcs,
        petsc_options_prefix=f"stokes_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    w_stokes = stokes_problem.solve()
    w.x.array[:] = w_stokes.x.array
    w.x.scatter_forward()

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    J = ufl.derivative(F, w)

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-10,
        "snes_atol": 1.0e-12,
        "snes_max_it": 30,
        "ksp_type": "gmres",
        "ksp_rtol": 1.0e-9,
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(
        F,
        w,
        bcs=bcs,
        J=J,
        petsc_options_prefix=f"ns_{n}_",
        petsc_options=opts,
    )

    t0 = time.perf_counter()
    w = problem.solve()
    solve_time = time.perf_counter() - t0
    w.x.scatter_forward()

    snes = problem.solver
    nonlinear_it = int(snes.getIterationNumber())
    ksp = snes.getKSP()
    linear_it = int(ksp.getIterationNumber())

    u_l2, p_l2 = _compute_errors(w, msh)

    return {
        "mesh": msh,
        "w": w,
        "u_l2": float(u_l2),
        "p_l2": float(p_l2),
        "solve_time": float(solve_time),
        "linear_iterations": linear_it,
        "nonlinear_iterations": [nonlinear_it],
        "ksp_type": str(ksp.getType()),
        "pc_type": str(ksp.getPC().getType()),
        "rtol": float(ksp.getTolerances()[0]),
        "mesh_resolution": int(n),
        "element_degree": 2,
    }


def solve(case_spec: dict) -> dict:
    _ = _diagnosis_and_method_cards()

    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    accuracy_target = 1.44e-4
    time_limit = 122.750
    target_fraction = 0.05
    target_budget = target_fraction * time_limit

    candidates = [28, 40, 56]
    best = None
    cumulative_runtime = 0.0

    for i, n in enumerate(candidates):
        result = _solve_on_mesh(n)
        cumulative_runtime += result["solve_time"]
        best = result

        if MPI.COMM_WORLD.rank == 0:
            print(
                f"[solver] n={n}, u_L2={result['u_l2']:.6e}, "
                f"p_L2={result['p_l2']:.6e}, solve_time={result['solve_time']:.3f}s, "
                f"cum_time={cumulative_runtime:.3f}s"
            )

        if result["u_l2"] > accuracy_target and i == len(candidates) - 1:
            break

        if cumulative_runtime >= target_budget:
            break

        if i < len(candidates) - 1:
            next_est = result["solve_time"] * (candidates[i + 1] / n) ** 3
            if cumulative_runtime + next_est > 0.92 * time_limit:
                break

    uh = best["w"].sub(0).collapse()
    uh.x.scatter_forward()
    vel_vals = _sample_function_on_grid(uh, best["mesh"], nx, ny, bbox)
    vel_mag = np.linalg.norm(vel_vals, axis=1).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": int(best["mesh_resolution"]),
        "element_degree": int(best["element_degree"]),
        "ksp_type": str(best["ksp_type"]),
        "pc_type": str(best["pc_type"]),
        "rtol": float(best["rtol"]),
        "iterations": int(best["linear_iterations"]),
        "nonlinear_iterations": [int(v) for v in best["nonlinear_iterations"]],
        "u_l2_error": float(best["u_l2"]),
        "p_l2_error": float(best["p_l2"]),
        "wall_time_sec": float(best["solve_time"]),
    }

    return {"u": vel_mag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
