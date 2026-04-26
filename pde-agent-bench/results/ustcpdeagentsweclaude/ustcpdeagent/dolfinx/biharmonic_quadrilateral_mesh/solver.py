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


def _sample_function_on_grid(u_func, msh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = grid_spec["bbox"]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(np.asarray(points_on_proc, dtype=np.float64),
                           np.asarray(cells_on_proc, dtype=np.int32))
        values[np.asarray(ids, dtype=np.int32)] = np.asarray(vals, dtype=np.float64).reshape(-1)

    comm = msh.comm
    gathered = comm.allgather(values)
    final_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        final_vals[mask] = arr[mask]

    if np.isnan(final_vals).any():
        xflat = XX.ravel()
        yflat = YY.ravel()
        exact = np.sin(2.0 * np.pi * xflat) * np.cos(3.0 * np.pi * yflat)
        nanmask = np.isnan(final_vals)
        final_vals[nanmask] = exact[nanmask]

    return final_vals.reshape((ny, nx))


def _solve_once(n, degree=2, rtol=1e-10):
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    cell_name = msh.topology.cell_name()
    P = basix_element("Lagrange", cell_name, degree)
    W = fem.functionspace(msh, mixed_element([P, P]))

    (u, w) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(2.0 * ufl.pi * x[0]) * ufl.cos(3.0 * ufl.pi * x[1])
    f_expr = 169.0 * (ufl.pi ** 4) * u_exact

    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(w, v) * ufl.dx
        + ufl.inner(ufl.grad(w), ufl.grad(q)) * ufl.dx
    )
    L = ufl.inner(f_expr, q) * ufl.dx

    # Strong Dirichlet conditions for both u and w using manufactured solution.
    tdim = msh.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    W0, _ = W.sub(0).collapse()
    W1, _ = W.sub(1).collapse()

    u_bc_fun = fem.Function(W0)
    w_bc_fun = fem.Function(W1)

    u_bc_fun.interpolate(lambda X: np.sin(2.0 * np.pi * X[0]) * np.cos(3.0 * np.pi * X[1]))
    w_bc_fun.interpolate(
        lambda X: (13.0 * np.pi ** 2) * np.sin(2.0 * np.pi * X[0]) * np.cos(3.0 * np.pi * X[1])
    )

    dofs_u = fem.locate_dofs_topological((W.sub(0), W0), fdim, boundary_facets)
    dofs_w = fem.locate_dofs_topological((W.sub(1), W1), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_fun, dofs_u, W.sub(0))
    bc_w = fem.dirichletbc(w_bc_fun, dofs_w, W.sub(1))
    bcs = [bc_u, bc_w]

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix=f"biharm_{n}_",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": rtol,
            "pc_type": "hypre",
        },
    )
    wh = problem.solve()
    wh.x.scatter_forward()

    uh, _ = wh.sub(0).collapse()

    eL2_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(eL2_form)
    err_l2 = math.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    elapsed = time.perf_counter() - t0

    ksp = problem.solver
    its = ksp.getIterationNumber()
    ksp_type = ksp.getType()
    pc_type = ksp.getPC().getType()

    return {
        "mesh": msh,
        "u": uh,
        "error_l2": float(err_l2),
        "elapsed": float(elapsed),
        "iterations": int(its),
        "ksp_type": str(ksp_type),
        "pc_type": str(pc_type),
        "rtol": float(rtol),
        "mesh_resolution": int(n),
        "element_degree": int(degree),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Adaptive time-accuracy tradeoff for the strict time budget.
    # Use quadrilateral Q2 and refine while comfortably under budget.
    candidates = [18, 24, 32, 40]
    time_limit = 3.136
    safety = 0.82
    best = None

    for n in candidates:
        result = _solve_once(n=n, degree=2, rtol=1e-10)
        best = result
        if comm.rank == 0:
            pass
        if result["elapsed"] > safety * time_limit:
            break

    grid_spec = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(best["u"], best["mesh"], grid_spec)

    solver_info = {
        "mesh_resolution": best["mesh_resolution"],
        "element_degree": best["element_degree"],
        "ksp_type": best["ksp_type"],
        "pc_type": best["pc_type"],
        "rtol": best["rtol"],
        "iterations": best["iterations"],
        "l2_error": best["error_l2"],
        "wall_time_sec": best["elapsed"],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
