import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def _exact_numpy(x):
    return np.sin(2.0 * np.pi * x[0]) * np.cos(3.0 * np.pi * x[1])


def _sample_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack((XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)))

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts2)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            missing = np.isnan(out).sum()
            raise RuntimeError(f"Failed to evaluate {missing} grid points.")
        return out.reshape(ny, nx)
    return None


def _solve_biharmonic(n, degree=2):
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )

    cell = domain.topology.cell_name()
    el_u = basix_element("Lagrange", cell, degree)
    el_w = basix_element("Lagrange", cell, degree)
    W = fem.functionspace(domain, basix_mixed_element([el_u, el_w]))

    V_u, _ = W.sub(0).collapse()
    V_w, _ = W.sub(1).collapse()

    (u, w) = ufl.TrialFunctions(W)
    (v, z) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(3.0 * ufl.pi * x[1])
    lap_u_exact = -((2.0 * ufl.pi) ** 2 + (3.0 * ufl.pi) ** 2) * u_exact
    f_expr = ((2.0 * ufl.pi) ** 2 + (3.0 * ufl.pi) ** 2) ** 2 * u_exact

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(w, v) * ufl.dx
        + ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx
    )
    L = ufl.inner(f_expr, z) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))

    u_bc_fun = fem.Function(V_u)
    u_bc_fun.interpolate(_exact_numpy)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_u), fdim, facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))

    w_bc_fun = fem.Function(V_w)
    w_bc_fun.interpolate(lambda X: -((2.0 * np.pi) ** 2 + (3.0 * np.pi) ** 2) * _exact_numpy(X))
    dofs_w = fem.locate_dofs_topological((W.sub(1), V_w), fdim, facets)
    bc_w = fem.dirichletbc(w_bc_fun, dofs_w, W.sub(1))

    opts = {"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1.0e-10}
    problem = petsc.LinearProblem(a, L, bcs=[bc_u, bc_w], petsc_options_prefix=f"bih_{n}_", petsc_options=opts)
    wh = problem.solve()
    wh.x.scatter_forward()

    uh = wh.sub(0).collapse()

    err_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * ufl.dx))
    err_L2 = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    ksp = problem.solver
    its = int(ksp.getIterationNumber())
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()

    return {
        "domain": domain,
        "uh": uh,
        "error_L2": float(err_L2),
        "iterations": its,
        "mesh_resolution": int(n),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1.0e-10,
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    budget = 7.485
    target = 0.85 * budget

    candidates = [20, 28, 36, 44, 52]
    timings = []
    best = None

    for n in candidates:
        ts = time.perf_counter()
        res = _solve_biharmonic(n, degree=2)
        te = time.perf_counter()
        timings.append(te - ts)
        best = res

        elapsed = te - t0
        avg = sum(timings) / len(timings)
        if elapsed + avg > target:
            break

    u_grid = _sample_on_grid(best["domain"], best["uh"], case_spec["output"]["grid"])

    if MPI.COMM_WORLD.rank == 0:
        return {
            "u": u_grid,
            "solver_info": {
                "mesh_resolution": best["mesh_resolution"],
                "element_degree": best["element_degree"],
                "ksp_type": best["ksp_type"],
                "pc_type": best["pc_type"],
                "rtol": best["rtol"],
                "iterations": best["iterations"],
                "verification_L2_error": best["error_L2"],
                "wall_time_sec": time.perf_counter() - t0,
            },
        }
    return {"u": None, "solver_info": {}}
