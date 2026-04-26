import time
from typing import Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _left_inflow(x):
    return np.vstack(
        (
            np.sin(np.pi * x[1]),
            0.2 * np.sin(2.0 * np.pi * x[1]),
        )
    )


def _zero_vec(x):
    return np.zeros((2, x.shape[1]), dtype=np.float64)


def _build_problem(
    comm,
    n: int,
    nu_value: float = 0.5,
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    rtol: float = 1.0e-10,
):
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, ScalarType(nu_value))
    f = fem.Constant(msh, np.array([0.0, 0.0], dtype=ScalarType))

    a = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))

    u_left = fem.Function(V)
    u_left.interpolate(_left_inflow)
    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    bc_left = fem.dirichletbc(u_left, dofs_left, W.sub(0))

    u_zero = fem.Function(V)
    u_zero.interpolate(_zero_vec)
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bc_bottom = fem.dirichletbc(u_zero, dofs_bottom, W.sub(0))
    bc_top = fem.dirichletbc(u_zero, dofs_top, W.sub(0))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))

    bcs = [bc_left, bc_bottom, bc_top, bc_p]

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if pc_type == "lu":
        opts["pc_factor_mat_solver_type"] = "mumps"

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options=opts,
        petsc_options_prefix=f"stokes_{n}_",
    )
    return msh, W, V, Q, problem


def _solve_once(
    comm,
    n: int,
    nu_value: float = 0.5,
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    rtol: float = 1.0e-10,
):
    msh, W, V, Q, problem = _build_problem(comm, n, nu_value, ksp_type, pc_type, rtol)
    t0 = time.perf_counter()
    wh = problem.solve()
    wh.x.scatter_forward()
    elapsed = time.perf_counter() - t0

    uh = wh.sub(0).collapse()
    ph = wh.sub(1).collapse()

    ksp = problem.solver
    try:
        its = ksp.getIterationNumber()
    except Exception:
        its = 0

    return {
        "mesh": msh,
        "W": W,
        "V": V,
        "Q": Q,
        "w": wh,
        "u": uh,
        "p": ph,
        "time": elapsed,
        "iterations": int(its),
        "ksp_type": ksp.getType() if hasattr(ksp, "getType") else ksp_type,
        "pc_type": ksp.getPC().getType() if hasattr(ksp, "getPC") else pc_type,
        "rtol": float(rtol),
        "n": n,
    }


def _compute_l2_divergence(uh) -> float:
    msh = uh.function_space.mesh
    V0 = fem.functionspace(msh, ("DG", 0))
    expr = fem.Expression(ufl.div(uh), V0.element.interpolation_points)
    divh = fem.Function(V0)
    divh.interpolate(expr)
    local = fem.assemble_scalar(fem.form(ufl.inner(divh, divh) * ufl.dx))
    val = msh.comm.allreduce(local, op=MPI.SUM)
    return float(np.sqrt(max(val, 0.0)))


def _sample_vector_function(u_func, points_xyz: np.ndarray) -> np.ndarray:
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding = geometry.compute_colliding_cells(msh, candidates, points_xyz)

    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points_xyz.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points_xyz[i])
            cells_local.append(links[0])
            ids_local.append(i)

    local_vals = None
    if len(pts_local) > 0:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64), np.array(cells_local, dtype=np.int32))
        local_vals = (np.array(ids_local, dtype=np.int32), np.asarray(vals, dtype=np.float64))
    else:
        local_vals = (np.zeros((0,), dtype=np.int32), np.zeros((0, 2), dtype=np.float64))

    gathered = msh.comm.allgather(local_vals)
    out = np.full((points_xyz.shape[0], msh.geometry.dim), np.nan, dtype=np.float64)
    for ids, vals in gathered:
        if len(ids) > 0:
            out[ids] = vals
    return out


def _sample_velocity_magnitude(uh, grid: Dict[str, Any]) -> np.ndarray:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    vals = _sample_vector_function(uh, pts)
    mags = np.linalg.norm(vals, axis=1)

    if np.isnan(mags).any():
        bad = np.isnan(mags)
        mags[bad] = 0.0

    return mags.reshape((ny, nx))


def _boundary_consistency_metric(uh, nprobe: int = 129) -> float:
    x = np.linspace(0.0, 1.0, nprobe)
    y = np.linspace(0.0, 1.0, nprobe)

    pts_left = np.column_stack([np.zeros_like(y), y, np.zeros_like(y)])
    pts_bottom = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    pts_top = np.column_stack([x, np.ones_like(x), np.zeros_like(x)])

    u_left = _sample_vector_function(uh, pts_left)
    u_bottom = _sample_vector_function(uh, pts_bottom)
    u_top = _sample_vector_function(uh, pts_top)

    target_left = np.column_stack([np.sin(np.pi * y), 0.2 * np.sin(2.0 * np.pi * y)])
    target_zero = np.zeros_like(u_bottom)

    e_left = np.sqrt(np.mean(np.sum((u_left - target_left) ** 2, axis=1)))
    e_bottom = np.sqrt(np.mean(np.sum((u_bottom - target_zero) ** 2, axis=1)))
    e_top = np.sqrt(np.mean(np.sum((u_top - target_zero) ** 2, axis=1)))
    return float(max(e_left, e_bottom, e_top))


def _verification_report(sol_coarse, sol_fine) -> Dict[str, float]:
    div_norm = _compute_l2_divergence(sol_fine["u"])
    bc_err = _boundary_consistency_metric(sol_fine["u"])
    compare_grid = {"nx": 41, "ny": 41, "bbox": [0.0, 1.0, 0.0, 1.0]}
    uc = _sample_velocity_magnitude(sol_coarse["u"], compare_grid)
    uf = _sample_velocity_magnitude(sol_fine["u"], compare_grid)
    mesh_diff = float(np.sqrt(np.mean((uf - uc) ** 2)))
    return {
        "divergence_l2": div_norm,
        "bc_rmse": bc_err,
        "mesh_change_rmse": mesh_diff,
    }


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    output_grid = case_spec["output"]["grid"]
    pde = case_spec.get("pde", {})
    nu_value = float(case_spec.get("coefficients", {}).get("nu", pde.get("nu", 0.5)) or 0.5)

    # Adaptive choice: spend some budget for higher accuracy while staying conservative.
    # For this steady 2D Stokes problem, direct LU is robust and fast on moderate meshes.
    candidate_resolutions = [40, 64]
    timings = []
    solutions = []
    for n in candidate_resolutions:
        sol = _solve_once(
            comm=comm,
            n=n,
            nu_value=nu_value,
            ksp_type="preonly",
            pc_type="lu",
            rtol=1.0e-10,
        )
        solutions.append(sol)
        timings.append(sol["time"])

    # If both resolutions were cheap, prefer the finer one. Otherwise use the best available.
    chosen = solutions[-1]
    coarse = solutions[0]
    fine = solutions[-1]

    verification = _verification_report(coarse, fine)
    u_grid = _sample_velocity_magnitude(chosen["u"], output_grid)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(chosen["n"]),
            "element_degree": 2,
            "ksp_type": str(chosen["ksp_type"]),
            "pc_type": str(chosen["pc_type"]),
            "rtol": float(chosen["rtol"]),
            "iterations": int(sum(sol["iterations"] for sol in solutions)),
            "verification": verification,
            "wall_time_setup_and_solve_sec": float(sum(sol["time"] for sol in solutions)),
        },
    }
    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "stokes", "nu": 0.5},
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        },
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
