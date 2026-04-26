import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS_CARD = "\n".join([
    "```DIAGNOSIS",
    "equation_type: convection_diffusion",
    "spatial_dim: 2",
    "domain_geometry: rectangle",
    "unknowns: scalar",
    "coupling: none",
    "linearity: linear",
    "time_dependence: steady",
    "stiffness: stiff",
    "dominant_physics: mixed",
    "peclet_or_reynolds: high",
    "solution_regularity: smooth",
    "bc_type: all_dirichlet",
    "special_notes: manufactured_solution",
    "```",
])

METHOD_CARD = "\n".join([
    "```METHOD",
    "spatial_method: fem",
    "element_or_basis: Lagrange_P2",
    "stabilization: supg",
    "time_method: none",
    "nonlinear_solver: none",
    "linear_solver: gmres",
    "preconditioner: ilu",
    "special_treatment: none",
    "pde_skill: convection_diffusion",
    "```",
])


def _exact_numpy(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _sample_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx, dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    values = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        values[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    gathered = msh.comm.allgather(values)
    merged = gathered[0].copy()
    for arr in gathered[1:]:
        mask = np.isnan(merged) & ~np.isnan(arr)
        merged[mask] = arr[mask]
    if np.isnan(merged).any():
        raise RuntimeError("Sampling failed for some points")
    return merged.reshape(ny, nx)


def _solve_once(comm, eps, beta_np, N, degree, ksp_type, pc_type, rtol, tau_scale):
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    uex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    beta = ufl.as_vector((ScalarType(beta_np[0]), ScalarType(beta_np[1])))
    f = -eps * ufl.div(ufl.grad(uex)) + ufl.dot(beta, ufl.grad(uex))

    uD = fem.Function(V)
    uD.interpolate(fem.Expression(uex, V.element.interpolation_points))

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(uD, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-30)
    pe = bnorm * h / (2.0 * eps + 1.0e-30)
    z = 2.0 * pe
    coth = (ufl.exp(z) + 1.0) / (ufl.exp(z) - 1.0 + 1.0e-30)
    tau_stream = h / (2.0 * bnorm + 1.0e-30) * (coth - 1.0 / (pe + 1.0e-30))
    tau = tau_scale * tau_stream

    Lu = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    supg_test = ufl.dot(beta, ufl.grad(v))

    a = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
        + tau * Lu * supg_test * ufl.dx
    )
    L = (f * v + tau * f * supg_test) * ufl.dx

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
        "ksp_atol": 1.0e-13,
        "ksp_max_it": 4000,
        "ksp_gmres_restart": 200,
    }

    try:
        problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"cdr_{N}_", petsc_options=opts)
        uh = problem.solve()
        uh.x.scatter_forward()
        its = int(problem.solver.getIterationNumber())
        used_ksp = ksp_type
        used_pc = pc_type
    except Exception:
        problem = petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix=f"cdrlu_{N}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        uh.x.scatter_forward()
        its = 1
        used_ksp = "preonly"
        used_pc = "lu"

    uexf = fem.Function(V)
    uexf.interpolate(fem.Expression(uex, V.element.interpolation_points))
    err_local = fem.assemble_scalar(fem.form((uh - uexf) * (uh - uexf) * ufl.dx))
    l2 = math.sqrt(max(msh.comm.allreduce(err_local, op=MPI.SUM), 0.0))
    return uh, l2, its, used_ksp, used_pc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]
    solver_cfg = case_spec.get("solver", {})

    eps = float(pde.get("epsilon", 0.005))
    beta_in = pde.get("beta", [20.0, 0.0])
    beta_np = np.array([float(beta_in[0]), float(beta_in[1])], dtype=np.float64)

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    degree = int(solver_cfg.get("element_degree", 2))
    ksp_type = str(solver_cfg.get("ksp_type", "gmres"))
    pc_type = str(solver_cfg.get("pc_type", "ilu"))
    rtol = float(solver_cfg.get("rtol", 1.0e-9))
    tau_scale = float(solver_cfg.get("upwind_parameter", 2.0))

    if "mesh_resolution" in solver_cfg:
        candidates = [int(solver_cfg["mesh_resolution"])]
    else:
        candidates = [72, 96, 128, 144]

    start = time.perf_counter()
    best = None
    for N in candidates:
        uh, l2, its, used_ksp, used_pc = _solve_once(
            comm, eps, beta_np, N, degree, ksp_type, pc_type, rtol, tau_scale
        )

        sx = min(nx, 96)
        sy = min(ny, 96)
        ug = _sample_function(uh, bbox, sx, sy)
        xs = np.linspace(bbox[0], bbox[1], sx)
        ys = np.linspace(bbox[2], bbox[3], sy)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        rmse = float(np.sqrt(np.mean((ug - _exact_numpy(xx, yy)) ** 2)))

        best = {
            "uh": uh,
            "N": N,
            "its": its,
            "ksp": used_ksp,
            "pc": used_pc,
            "l2": l2,
            "rmse": rmse,
        }

        elapsed = time.perf_counter() - start
        if "mesh_resolution" in solver_cfg:
            break
        if elapsed > 1.45:
            break
        if rmse < 1.5e-4:
            continue

    u_grid = _sample_function(best["uh"], bbox, nx, ny)
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best["N"]),
            "element_degree": int(degree),
            "ksp_type": str(best["ksp"]),
            "pc_type": str(best["pc"]),
            "rtol": float(rtol),
            "iterations": int(best["its"]),
            "l2_error_verification": float(best["l2"]),
            "grid_rmse_verification": float(best["rmse"]),
        },
    }


def _run_self_test():
    case = {
        "pde": {"epsilon": 0.005, "beta": [20.0, 0.0]},
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    out = solve(case)
    wall = time.perf_counter() - t0
    xs = np.linspace(0.0, 1.0, 128)
    ys = np.linspace(0.0, 1.0, 128)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    uex = _exact_numpy(xx, yy)
    err = float(np.sqrt(np.mean((out["u"] - uex) ** 2)))
    if MPI.COMM_WORLD.rank == 0:
        print(DIAGNOSIS_CARD)
        print(METHOD_CARD)
        print(f"GRID_RMSE: {err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(out["solver_info"])


if __name__ == "__main__":
    _run_self_test()
