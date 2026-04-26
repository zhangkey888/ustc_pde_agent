import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _manufactured_u_expr(x):
    return np.exp(6.0 * x[0]) * np.sin(np.pi * x[1])


def _probe_function(u_func, points):
    msh = u_func.function_space.mesh
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(msh, candidates, points)
    values = np.full(points.shape[0], np.nan, dtype=np.float64)
    pts_local = []
    cells_local = []
    ids_local = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_local.append(points[i])
            cells_local.append(links[0])
            ids_local.append(i)
    if pts_local:
        vals = u_func.eval(np.array(pts_local, dtype=np.float64),
                           np.array(cells_local, dtype=np.int32))
        values[np.array(ids_local, dtype=np.int32)] = np.asarray(vals).reshape(-1).real
    return values


def _build_and_solve(nx, degree, ksp_type="cg", pc_type="hypre", rtol=1.0e-10):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, nx, nx, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = ufl.exp(6.0 * x[0]) * ufl.sin(ufl.pi * x[1])
    kappa = ScalarType(1.0)
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.full(X.shape[1], True, dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_manufactured_u_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    opts = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": rtol,
    }
    if ksp_type == "cg" and pc_type == "hypre":
        opts["pc_hypre_type"] = "boomeramg"

    uh = fem.Function(V)
    start = time.perf_counter()
    try:
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix="poisson_",
            petsc_options=opts,
        )
        uh = problem.solve()
        ksp = problem.solver
    except Exception:
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix="poisson_fallback_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_rtol": rtol,
            },
        )
        uh = problem.solve()
        ksp = problem.solver
        ksp_type = "preonly"
        pc_type = "lu"
    solve_time = time.perf_counter() - start
    uh.x.scatter_forward()

    u_ex = fem.Function(V)
    u_ex.interpolate(_manufactured_u_expr)
    err_L2 = math.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form((uh - u_ex) ** 2 * ufl.dx)),
            op=MPI.SUM,
        )
    )

    return {
        "mesh": msh,
        "V": V,
        "uh": uh,
        "err_L2": err_L2,
        "solve_time": solve_time,
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
    }


def solve(case_spec: dict) -> dict:
    t0 = time.perf_counter()
    budget = 2.846
    target_err = 1.40e-03

    # Adaptive accuracy strategy: escalate while comfortably under budget.
    candidates = [(24, 1), (32, 1), (48, 1), (24, 2), (32, 2), (40, 2)]
    best = None

    for nx, degree in candidates:
        remaining = budget - (time.perf_counter() - t0)
        if remaining <= 0.35 and best is not None:
            break
        result = _build_and_solve(nx, degree)
        if best is None or result["err_L2"] < best["err_L2"]:
            best = result
        elapsed = time.perf_counter() - t0
        if result["err_L2"] <= target_err and elapsed > 0.7 * budget:
            break
        if elapsed > 0.85 * budget:
            break

    # If still very fast and accurate, try one more refinement.
    elapsed = time.perf_counter() - t0
    if best is not None and best["err_L2"] <= target_err and elapsed < 0.6 * budget:
        try:
            extra = _build_and_solve(min(best["mesh_resolution"] + 8, 56), max(best["element_degree"], 2))
            if extra["err_L2"] < best["err_L2"] and (time.perf_counter() - t0) < budget:
                best = extra
        except Exception:
            pass

    msh = best["mesh"]
    uh = best["uh"]

    grid = case_spec["output"]["grid"]
    nxg = int(grid["nx"])
    nyg = int(grid["ny"])
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nxg)
    ys = np.linspace(bbox[2], bbox[3], nyg)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nxg * nyg, dtype=np.float64)])
    vals = _probe_function(uh, pts)
    u_grid = vals.reshape(nyg, nxg)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": int(best["mesh_resolution"]),
            "element_degree": int(best["element_degree"]),
            "ksp_type": str(best["ksp_type"]),
            "pc_type": str(best["pc_type"]),
            "rtol": float(best["rtol"]),
            "iterations": int(best["iterations"]),
            "l2_error": float(best["err_L2"]),
            "wall_time_sec": float(time.perf_counter() - t0),
        },
    }
