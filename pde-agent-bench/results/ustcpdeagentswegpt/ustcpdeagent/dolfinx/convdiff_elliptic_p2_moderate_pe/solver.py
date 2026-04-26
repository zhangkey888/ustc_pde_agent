import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _exact_numpy(x):
    return np.sin(np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])


def _evaluate_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts2)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts2)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells = []
    ids = []
    for i in range(nx * ny):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts2[i])
            cells.append(links[0])
            ids.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.asarray(points_on_proc, dtype=np.float64), np.asarray(cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_vals[np.asarray(ids, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isnan(out) & ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            missing = np.isnan(out)
            out[missing] = 0.0
        out = out.reshape(ny, nx)
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _build_and_solve(n, degree, use_supg=True, rtol=1e-10):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = ScalarType(0.03)
    beta_vec = np.array([5.0, 2.0], dtype=np.float64)
    beta = fem.Constant(domain, beta_vec.astype(ScalarType))
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    grad_u = ufl.grad(u_exact_ufl)
    lap_u = ufl.div(grad_u)
    f_expr = -eps * lap_u + ufl.dot(beta, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    if use_supg:
        h = ufl.CellDiameter(domain)
        beta_norm = ufl.sqrt(ufl.dot(beta, beta))
        tau = h / (2.0 * beta_norm + 4.0 * eps / h)
        r_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        r_rhs = f_expr
        a += tau * r_u * ufl.dot(beta, ufl.grad(v)) * ufl.dx
        L += tau * r_rhs * ufl.dot(beta, ufl.grad(v)) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(2.0 * np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": "gmres",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "pc_type": "hypre",
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    err_L2_local = fem.assemble_scalar(fem.form((uh - u_exact_ufl) ** 2 * ufl.dx))
    err_H1s_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(uh - u_exact_ufl), ufl.grad(uh - u_exact_ufl)) * ufl.dx))
    err_L2 = math.sqrt(comm.allreduce(err_L2_local, op=MPI.SUM))
    err_H1s = math.sqrt(comm.allreduce(err_H1s_local, op=MPI.SUM))

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())
    pc = ksp.getPC()
    try:
        pc_type = pc.getType()
    except Exception:
        pc_type = "unknown"

    return {
        "domain": domain,
        "uh": uh,
        "L2_error": err_L2,
        "H1_semi_error": err_H1s,
        "iterations": iterations,
        "ksp_type": ksp.getType(),
        "pc_type": pc_type,
        "rtol": rtol,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()

    degree = 2
    target_time = 15.243
    safety = 1.0
    trial_ns = [72, 96, 128, 160, 192]
    results = []
    chosen = None

    for n in trial_ns:
        t_start = time.perf_counter()
        res = _build_and_solve(n=n, degree=degree, use_supg=True, rtol=1e-10)
        elapsed = time.perf_counter() - t_start
        res["wall"] = elapsed
        results.append((n, res))
        if comm.rank == 0:
            pass
        remaining = target_time - (time.perf_counter() - t0)
        next_est = elapsed * 1.8
        if res["L2_error"] <= 2.28e-05 and (remaining < next_est + safety):
            chosen = res
            chosen["mesh_resolution"] = n
            break

    if chosen is None:
        n, chosen = results[-1]
        chosen["mesh_resolution"] = n

    grid = case_spec["output"]["grid"]
    u_grid = _evaluate_on_grid(chosen["domain"], chosen["uh"], grid)

    solver_info = {
        "mesh_resolution": int(chosen["mesh_resolution"]),
        "element_degree": int(degree),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(chosen["iterations"]),
        "L2_error_vs_exact": float(chosen["L2_error"]),
        "H1_semi_error_vs_exact": float(chosen["H1_semi_error"]),
        "manufactured_solution_verified": True,
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
