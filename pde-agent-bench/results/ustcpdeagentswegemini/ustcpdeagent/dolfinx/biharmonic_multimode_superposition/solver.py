import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _u_exact_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 0.5 * np.sin(2 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])


def _f_numpy(x):
    return (
        4 * np.pi**4 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        + 84 * np.pi**4 * 0.5 * np.sin(2 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
    )


def _sample_function(u_func, msh, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        vals = np.asarray(vals).reshape(-1)
        local_vals[np.array(eval_ids, dtype=np.int32)] = vals.real

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        out = out.reshape(ny, nx)
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _build_bc_function(V):
    u_bc = fem.Function(V)
    u_bc.interpolate(
        lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 0.5 * np.sin(2 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
    )
    return u_bc


def _build_w_bc_function(V):
    # w = -Δu_exact = 2π² sin(πx)sin(πy) + 6.5π² sin(2πx)sin(3πy)
    w_bc = fem.Function(V)
    w_bc.interpolate(
        lambda x: 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        + 6.5 * np.pi**2 * np.sin(2 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
    )
    return w_bc


def _solve_poisson(V, a, L, bc, prefix, ksp_type, pc_type, rtol):
    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    its = problem.solver.getIterationNumber()
    return uh, its


def _run_single(n, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc_fun = _build_bc_function(V)
    w_bc_fun = _build_w_bc_function(V)
    bc_u = fem.dirichletbc(u_bc_fun, dofs)
    bc_w = fem.dirichletbc(w_bc_fun, dofs)

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_expr = (
        4 * ufl.pi**4 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        + 42 * ufl.pi**4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    )

    a1 = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    L1 = ufl.inner(f_expr, v) * ufl.dx
    w_h, its1 = _solve_poisson(V, a1, L1, bc_w, f"bih_w_{n}_", ksp_type, pc_type, rtol)

    a2 = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    L2 = ufl.inner(w_h, v) * ufl.dx
    u_h, its2 = _solve_poisson(V, a2, L2, bc_u, f"bih_u_{n}_", ksp_type, pc_type, rtol)

    u_ex = fem.Function(V)
    u_ex.interpolate(lambda xx: _u_exact_numpy(xx))
    err_form = fem.form((u_h - u_ex) ** 2 * ufl.dx)
    ref_form = fem.form((u_ex) ** 2 * ufl.dx)
    err_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    ref_l2 = math.sqrt(comm.allreduce(fem.assemble_scalar(ref_form), op=MPI.SUM))
    rel_l2 = err_l2 / ref_l2 if ref_l2 > 0 else err_l2

    return {
        "mesh": msh,
        "V": V,
        "u_h": u_h,
        "iterations": int(its1 + its2),
        "rel_l2": float(rel_l2),
        "abs_l2": float(err_l2),
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    start = time.perf_counter()

    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]

    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Adaptive accuracy/time trade-off: try progressively finer meshes while respecting budget.
    # Conservative candidate set to stay under ~9.5s in typical single-rank evaluation.
    candidates = [40, 56, 72, 88]
    best = None
    budget = 8.8

    for n in candidates:
        t0 = time.perf_counter()
        result = _run_single(n, degree, ksp_type, pc_type, rtol)
        elapsed_now = time.perf_counter() - start
        if best is None or result["rel_l2"] < best["rel_l2"]:
            best = result
        # stop refining if already accurate enough and time is getting tight
        if elapsed_now > budget:
            break
        # simple predictive guard
        step_time = time.perf_counter() - t0
        if elapsed_now + 1.6 * step_time > budget and result["rel_l2"] < 5e-4:
            break

    u_grid = _sample_function(best["u_h"], best["mesh"], nx, ny, bbox)

    solver_info = {
        "mesh_resolution": int(candidates[0] if best is None else best["mesh"].topology.index_map(2).size_local**0),
        "element_degree": int(degree),
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": float(rtol),
        "iterations": int(best["iterations"]),
        "verification_rel_l2": float(best["rel_l2"]),
        "verification_abs_l2": float(best["abs_l2"]),
        "wall_time_sec": float(time.perf_counter() - start),
    }
    # Report actual mesh resolution as subdivisions per direction
    num_cells_local = best["mesh"].topology.index_map(best["mesh"].topology.dim).size_local
    if comm.rank == 0:
        approx_n = int(round((num_cells_local * comm.size / 2) ** 0.5))
    else:
        approx_n = None
    approx_n = comm.bcast(approx_n, root=0)
    solver_info["mesh_resolution"] = approx_n

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
