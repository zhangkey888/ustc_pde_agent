import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _make_exact_velocity(x):
    return np.vstack(
        [
            x[0] ** 2 * (1.0 - x[0]) ** 2 * (1.0 - 2.0 * x[1]),
            -2.0 * x[0] * (1.0 - x[0]) * (1.0 - 2.0 * x[0]) * x[1] * (1.0 - x[1]),
        ]
    )


def _sample_function_on_points(domain, func, points):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)

    value_shape = func.function_space.element.value_shape; value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    local_vals = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    if len(pts_local) > 0:
        vals = func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        vals = np.asarray(vals, dtype=np.float64).reshape(len(pts_local), value_size)
        local_vals[np.array(ids_local, dtype=np.int32), :] = vals

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full_like(gathered[0], np.nan)
        for arr in gathered:
            mask = np.isnan(out[:, 0]) & ~np.isnan(arr[:, 0])
            out[mask, :] = arr[mask, :]
        return out
    return None


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    t0 = time.time()

    # Adaptive-but-safe defaults chosen to satisfy accuracy well within the time budget.
    mesh_resolution = int(case_spec.get("solver", {}).get("mesh_resolution", 72))
    degree_u = int(case_spec.get("solver", {}).get("degree_u", 3))
    degree_p = int(case_spec.get("solver", {}).get("degree_p", max(1, degree_u - 1)))
    newton_rtol = float(case_spec.get("solver", {}).get("newton_rtol", 1.0e-10))
    newton_max_it = int(case_spec.get("solver", {}).get("newton_max_it", 30))

    domain = mesh.create_unit_square(
        comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = domain.geometry.dim
    cell_name = domain.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", cell_name, degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(domain)
    nu = ScalarType(0.22)

    u_exact_ufl = ufl.as_vector(
        [
            x[0] ** 2 * (1 - x[0]) ** 2 * (1 - 2 * x[1]),
            -2 * x[0] * (1 - x[0]) * (1 - 2 * x[0]) * x[1] * (1 - x[1]),
        ]
    )
    p_exact_ufl = x[0] + x[1]

    f_ufl = ufl.as_vector(ufl.grad(u_exact_ufl) * u_exact_ufl - nu * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    w = fem.Function(W)
    vq = ufl.TestFunctions(W)
    (v, q) = vq
    u, p = ufl.split(w)

    alpha_c = fem.Constant(domain, ScalarType(1.0))
    F = (
        (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) + alpha_c * ufl.inner(ufl.grad(u) * u, v) - p * ufl.div(v) + ufl.div(u) * q - ufl.inner(f_ufl, v)) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Velocity Dirichlet BC from exact solution on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.interpolate(_make_exact_velocity)
    vel_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, vel_dofs, W.sub(0))

    # Pressure pinning for uniqueness
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess from exact boundary extension
    w.x.array[:] = 0.0
    w.sub(0).interpolate(_make_exact_velocity)
    w.x.scatter_forward()

    # Continuation on convection strength for robustness; final step is full NS
    nonlinear_iterations = []
    total_linear_iterations = 0
    alphas = [0.0, 0.35, 0.7, 1.0]

    for i, alpha in enumerate(alphas):
        alpha_c.value = ScalarType(alpha)
        J_alpha = ufl.derivative(F, w)

        opts = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": newton_rtol,
            "snes_atol": 1.0e-12,
            "snes_stol": 1.0e-12,
            "snes_max_it": newton_max_it,
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-9,
            "pc_type": "lu" if mesh_resolution <= 40 else "ilu",
        }

        problem = petsc.NonlinearProblem(
            F,
            w,
            bcs=bcs,
            J=J_alpha,
            petsc_options_prefix=f"ns_{i}_",
            petsc_options=opts,
        )
        problem.solve()
        w.x.scatter_forward()

        snes = problem.solver
        try:
            nonlinear_iterations.append(int(snes.getIterationNumber()))
        except Exception:
            nonlinear_iterations.append(0)
        try:
            total_linear_iterations += int(snes.getLinearSolveIterations())
        except Exception:
            pass

    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()

    # Accuracy verification against manufactured solution
    u_exact_fun = fem.Function(V)
    u_exact_fun.interpolate(_make_exact_velocity)
    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact_fun.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact_fun, u_exact_fun) * ufl.dx))
    l2_err = np.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = np.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_ref, 1e-14)

    # Sample velocity magnitude on specified output grid
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    sampled = _sample_function_on_points(domain, uh, pts)
    if rank == 0:
        velmag = np.linalg.norm(sampled, axis=1).reshape(ny, nx)
    else:
        velmag = None

    wall_time = time.time() - t0
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "gmres",
        "pc_type": "lu" if mesh_resolution <= 40 else "ilu",
        "rtol": 1.0e-9,
        "iterations": int(total_linear_iterations),
        "nonlinear_iterations": nonlinear_iterations,
        "verification_l2_error": float(l2_err),
        "verification_relative_l2_error": float(rel_l2_err),
        "wall_time_sec": float(wall_time),
    }

    if rank == 0:
        return {"u": velmag, "solver_info": solver_info}
    return {"u": np.zeros((ny, nx), dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 32, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "solver": {"mesh_resolution": 32, "degree_u": 3, "degree_p": 2},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
