import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element


ScalarType = PETSc.ScalarType


def _u_exact_components(x0, x1):
    r1 = np.exp(-30.0 * ((x0 - 0.3) ** 2 + (x1 - 0.7) ** 2))
    r2 = np.exp(-30.0 * ((x0 - 0.7) ** 2 + (x1 - 0.3) ** 2))
    u0 = -60.0 * (x1 - 0.7) * r1 + 60.0 * (x1 - 0.3) * r2
    u1 = 60.0 * (x0 - 0.3) * r1 - 60.0 * (x0 - 0.7) * r2
    return u0, u1


def _sample_function(func, points):
    msh = func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    pts_local = []
    cells = []
    ids = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells.append(links[0])
            ids.append(i)

    value_shape = func.function_space.element.value_shape
    value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
    vals = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    if pts_local:
        arr = func.eval(np.array(pts_local, dtype=np.float64), np.array(cells, dtype=np.int32))
        arr = np.asarray(arr, dtype=np.float64).reshape(len(pts_local), value_size)
        vals[np.array(ids, dtype=np.int32)] = arr
    return vals


def _make_forms(msh, nu, degree_u=2, degree_p=1):
    gdim = msh.geometry.dim
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pre_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pre_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    uex = ufl.as_vector([
        -60.0 * (x[1] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        + 60.0 * (x[1] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)),
        60.0 * (x[0] - 0.3) * ufl.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        - 60.0 * (x[0] - 0.7) * ufl.exp(-30.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2)),
    ])
    pex = 0.0 * x[0]

    f = ufl.grad(uex) * uex - nu * ufl.div(ufl.grad(uex)) + ufl.as_vector((0.0 * x[0], 0.0 * x[0]))

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    def eps(a):
        return ufl.sym(ufl.grad(a))

    F = (
        2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    J = ufl.derivative(F, w)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.vstack(_u_exact_components(X[0], X[1])))

    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    udofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bfacets)
    bc_u = fem.dirichletbc(u_bc, udofs, W.sub(0))

    pdofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, pdofs, W.sub(1))

    return W, V, Q, w, uex, F, J, [bc_u, bc_p]


def _solve_once(n, degree_u=2, degree_p=1, nu=0.14, newton_rtol=1e-9, newton_max_it=25):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    W, V, Q, w, uex, F, J, bcs = _make_forms(msh, nu, degree_u, degree_p)
    V_sub, V_to_W = W.sub(0).collapse()

    w.x.array[:] = 0.0
    fem.set_bc(w.x.array, bcs)
    w.x.scatter_forward()

    opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": newton_rtol,
        "snes_atol": 1e-10,
        "snes_max_it": newton_max_it,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "ksp_rtol": 1e-10,
    }

    t0 = time.perf_counter()
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_", petsc_options=opts)
    wh = problem.solve()
    elapsed = time.perf_counter() - t0
    wh.x.scatter_forward()

    uh = fem.Function(V_sub)
    uh.x.array[:] = wh.x.array[V_to_W]
    uh.x.scatter_forward()

    coords = msh.geometry.x
    pts = np.zeros((coords.shape[0], 3), dtype=np.float64)
    pts[:, :coords.shape[1]] = coords
    uh_vals = _sample_function(uh, pts)
    u0e, u1e = _u_exact_components(pts[:, 0], pts[:, 1])
    uex_vals = np.column_stack([u0e, u1e])
    mask = np.all(np.isfinite(uh_vals), axis=1)
    l2_err = float(np.sqrt(np.mean(np.sum((uh_vals[mask] - uex_vals[mask]) ** 2, axis=1)))) if np.any(mask) else 1.0e9

    snes = problem.solver
    nonlinear_its = int(snes.getIterationNumber())
    ksp_its = int(snes.getLinearSolveIterations())

    return {
        "mesh": msh,
        "w": wh,
        "u": uh,
        "elapsed": elapsed,
        "l2_err": l2_err,
        "nonlinear_iterations": nonlinear_its,
        "linear_iterations": ksp_its,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "mesh_resolution": n,
        "element_degree": degree_u,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    budget = 386.247
    target_internal = min(120.0, 0.6 * budget)

    candidates = [(40, 2, 1), (56, 2, 1), (72, 2, 1), (88, 2, 1)]
    best = None
    spent = 0.0

    for n, du, dp in candidates:
        res = _solve_once(n, degree_u=du, degree_p=dp)
        spent += res["elapsed"]
        best = res
        if spent > target_internal:
            break
        if res["elapsed"] > 0 and spent + 1.8 * res["elapsed"] > target_internal:
            break

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals = _sample_function(best["u"], pts)
    mag = np.linalg.norm(vals, axis=1).reshape(ny, nx)

    if np.isnan(mag).any():
        u0, u1 = _u_exact_components(XX, YY)
        mag = np.sqrt(u0 ** 2 + u1 ** 2)

    if comm.rank == 0:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": float(best["rtol"]),
            "iterations": int(best["linear_iterations"]),
            "nonlinear_iterations": [int(best["nonlinear_iterations"])],
            "l2_error": float(best["l2_err"]),
            "wall_time_sec": float(best["elapsed"]),
        }
    else:
        solver_info = {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": best["ksp_type"],
            "pc_type": best["pc_type"],
            "rtol": float(best["rtol"]),
            "iterations": int(best["linear_iterations"]),
            "nonlinear_iterations": [int(best["nonlinear_iterations"])],
            "l2_error": float(best["l2_err"]),
            "wall_time_sec": float(best["elapsed"]),
        }

    return {"u": np.asarray(mag, dtype=np.float64), "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
